import io
import json
import logging
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from libro_tts.catalog import CATALOG, MODEL_CONFIG_TYPE_ADAPTERS, canonical_model_type_for_key, get_model_spec
from libro_tts.cli import main
from libro_tts.diagnostics import TailCapture
from libro_tts import env
from libro_tts.runtime import (
    ReferenceContext, RuntimeOptions, _generate_chunk_audio, _populate_missing_ref_text,
    _resolve_reference_text, generate_tts, process_batch_dir,
)
from libro_tts.store import ModelStore


LOGGER = logging.getLogger("test.resources")


class ExecutionPolicyTests(unittest.TestCase):
    def test_cli_execution_flags_and_defaults(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            source = root / "input.txt"
            source.write_text("hello")
            for name in ("configure_local_cache_environment", "validate_runtime_environment",
                         "validate_mlx_backend_preflight", "validate_tts_model_runtime_support"):
                stack.enter_context(patch(f"libro_tts.cli.{name}"))
            stack.enter_context(patch("libro_tts.cli.prepare_runtime_dirs", return_value={"models_dir": root}))
            stack.enter_context(patch("libro_tts.store.ModelStore"))
            process = stack.enter_context(patch("libro_tts.runtime.process_single_file"))
            for flags, workers, batch in (([], 2, 12), (["--serial"], 1, 12),
                                          (["--workers", "1"], 1, 12),
                                          (["--workers", "3", "--parallel-batch-size", "5"], 3, 5)):
                with self.subTest(flags=flags):
                    self.assertEqual(main([str(source), *flags]), 0)
                    options = process.call_args.kwargs["options"]
                    self.assertEqual((options.parallel_workers, options.parallel_max_chunks), (workers, batch))

    def test_bad_execution_flags_fail_before_environment_work(self):
        with patch("libro_tts.cli.validate_runtime_environment") as validate, redirect_stderr(io.StringIO()):
            for flags in (["--serial", "--workers", "2"], ["--workers", "0"], ["--workers", "-2"],
                          ["--parallel-batch-size", "0"], ["--workers", "1.5"]):
                with self.subTest(flags=flags), self.assertRaises(SystemExit) as raised:
                    main(["input.txt", *flags])
                self.assertEqual(raised.exception.code, 2)
            validate.assert_not_called()

    def test_parallel_success_does_not_load_parent_and_serial_or_fallback_loads_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = RuntimeOptions("soprano", None, None, None, None, None, True, True, stream_wav=False)
            for mode in ("parallel", "serial", "fallback"):
                with self.subTest(mode=mode), ExitStack() as stack:
                    stack.enter_context(patch("libro_tts.runtime._resolve_model_reference", return_value=(str(root), get_model_spec("soprano"))))
                    stack.enter_context(patch("libro_tts.runtime.prepare_model_assets"))
                    stack.enter_context(patch("libro_tts.runtime.validate_model_load_preflight"))
                    stack.enter_context(patch("libro_tts.runtime._chunk_text_for_generation", return_value=["one", "two", "three"]))
                    load = stack.enter_context(patch("libro_tts.runtime._load_runtime_model"))
                    parallel = stack.enter_context(patch("libro_tts.runtime._generate_audio_parallel", return_value=([np.array([0.1])], 24000)))
                    serial = stack.enter_context(patch("libro_tts.runtime._generate_audio_serial", return_value=([np.array([0.1])], 24000)))
                    stack.enter_context(patch("libro_tts.runtime._write_output_audio"))
                    if mode == "fallback":
                        parallel.side_effect = RuntimeError("worker failed")
                    generate_tts(text="hello", output_prefix=str(root / "out.wav"),
                        model_store=SimpleNamespace(root_dir=root),
                        options=replace(options, parallel_workers=1) if mode == "serial" else options, logger=LOGGER)
                    self.assertEqual(load.call_count, 0 if mode == "parallel" else 1)
                    self.assertEqual(serial.call_count, 0 if mode == "parallel" else 1)
                    self.assertEqual(parallel.call_count, 0 if mode == "serial" else 1)

    def test_catalog_capabilities_keep_prompt_and_parallel_guards(self):
        self.assertEqual({key for key, spec in CATALOG.items() if spec.voice_mode == "prompt"},
                         {"csm", "dia", "spark", "chatterbox"})
        self.assertEqual({key for key, spec in CATALOG.items() if spec.requires_ref_text}, {"csm", "dia", "spark"})
        self.assertEqual({key for key, spec in CATALOG.items() if not spec.parallel_safe}, {"csm", "dia"})


class DiagnosticCaptureTests(unittest.TestCase):
    def test_many_and_oversized_writes_keep_only_bounded_tail(self):
        with TailCapture(limit=32) as sink:
            for _ in range(1000):
                self.assertEqual(sink.write("prefix"), 6)
                self.assertLessEqual(len(sink.getvalue()), 32)
            self.assertEqual(sink.write("X" * 100000 + "final context"), 100013)
            self.assertEqual(sink.getvalue(), "X" * 19 + "final context")
        with self.assertRaises(ValueError):
            sink.write("closed")

    def test_failure_keeps_recent_output_and_success_stays_quiet(self):
        def failed_generate(**_kwargs):
            print("discarded prefix" + "X" * 20000)
            print("last error context")
            raise RuntimeError("generation broke")
            yield  # Make failure occur while consuming the stream.
        model = SimpleNamespace(sample_rate=24000, generate=failed_generate)
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out), self.assertRaisesRegex(RuntimeError, "generation broke") as raised:
            _generate_chunk_audio(model=model, kwargs_template={}, text_chunk="hello", suppress_model_output=True)
        message = str(raised.exception)
        self.assertIn("last error context", message)
        self.assertNotIn("discarded prefix", message)
        self.assertLess(len(message), 8300)
        self.assertEqual(out.getvalue(), "")
        def successful_generate(**_kwargs):
            print("noisy success")
            yield SimpleNamespace(audio=np.array([0.1]), sample_rate=24000)
        model.generate = successful_generate
        with redirect_stdout(out):
            audio, rate = _generate_chunk_audio(model=model, kwargs_template={}, text_chunk="hello", suppress_model_output=True)
        self.assertEqual(rate, 24000)
        self.assertEqual(audio.size, 1)
        self.assertEqual(out.getvalue(), "")


class BatchReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.audio = self.root / "voice.wav"
        self.audio.write_bytes(b"fixture")
        self.text = self.audio.with_suffix(".txt")
        self.context = ReferenceContext()

    def resolve(self, *, model="dia", context=None, explicit=None, store=None):
        kwargs = {"ref_audio": str(self.audio)}
        if explicit is not None:
            kwargs["ref_text"] = explicit
        resolved = _populate_missing_ref_text(model_key=model, kwargs=kwargs, logger=LOGGER,
            verbose=False, model_store=store, reference_context=context or self.context)
        self.assertEqual(kwargs["ref_text"], resolved.text)
        return resolved

    def test_unchanged_reference_reuses_file_text_and_preserves_provenance(self):
        self.text.write_text("reference words")
        with patch("libro_tts.runtime._resolve_reference_text", wraps=_resolve_reference_text) as read:
            first = self.resolve()
            self.assertIs(self.resolve(), first)
            self.assertEqual(first.source, "file")
            read.assert_called_once()
            self.resolve(context=ReferenceContext())
            self.assertEqual(read.call_count, 2)

    def test_file_audio_store_and_model_changes_invalidate_reuse(self):
        self.text.write_text("first")
        with patch("libro_tts.runtime._resolve_reference_text", wraps=_resolve_reference_text) as read:
            self.resolve()
            self.text.write_text("changed text")
            self.assertEqual(self.resolve().text, "changed text")
            self.audio.write_bytes(b"changed reference audio")
            self.resolve()
            self.resolve(model="csm")
            self.resolve(store=SimpleNamespace(root_dir=self.root / "other-store"))
            self.assertEqual(read.call_count, 5)

    def test_stt_reused_but_new_sibling_or_explicit_text_takes_precedence(self):
        with patch("libro_tts.runtime._auto_transcribe_reference_text", return_value="transcribed") as stt:
            self.assertEqual(self.resolve().source, "stt")
            self.resolve()
            stt.assert_called_once()
            self.text.write_text("new sibling")
            self.assertEqual(self.resolve().source, "file")
            self.assertEqual(self.resolve(explicit="explicit").source, "explicit")
            stt.assert_called_once()

    def test_failed_transcription_is_not_cached(self):
        with patch("libro_tts.runtime._auto_transcribe_reference_text", side_effect=[RuntimeError("failed"), "recovered"]) as stt:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.resolve()
            self.assertEqual(self.resolve().text, "recovered")
            self.assertEqual(stt.call_count, 2)

    def test_batch_passes_one_context_and_new_run_gets_a_fresh_context(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        for name in ("one.txt", "two.txt"):
            (inputs / name).write_text("hello")
        options = RuntimeOptions("dia", None, None, None, None, None, True, True)
        with patch("libro_tts.runtime.generate_tts") as generate:
            contexts = []
            for _ in range(2):
                process_batch_dir(input_dir=inputs, output_dir=self.root / "output", model_store=Mock(), options=options, logger=LOGGER)
                calls = generate.call_args_list[-2:]
                self.assertIs(calls[0].kwargs["reference_context"], calls[1].kwargs["reference_context"])
                contexts.append(calls[0].kwargs["reference_context"])
            self.assertIsNot(contexts[0], contexts[1])


class MetadataAndProbeTests(unittest.TestCase):
    def test_known_metadata_adapters_and_canonical_spellings_preserve_other_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ModelStore(root)
            for key in CATALOG:
                expected = canonical_model_type_for_key(key)
                for label in (None, expected, f" {expected.upper()} ", *MODEL_CONFIG_TYPE_ADAPTERS.get(key, ())):
                    with self.subTest(key=key, label=label):
                        config = root / "config.json"
                        config.write_text(json.dumps({"model_type": label, "unrelated": {"keep": True}}))
                        store._ensure_model_type_metadata(get_model_spec(key), root)
                        self.assertEqual(json.loads(config.read_text()), {"model_type": expected, "unrelated": {"keep": True}})

    def test_unknown_or_nonstring_architectures_are_rejected_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ModelStore(root)
            for label in ("llama", "unrelated", [], 42):
                with self.subTest(label=label):
                    config = root / "config.json"
                    original = json.dumps({"model_type": label})
                    config.write_text(original)
                    with self.assertRaisesRegex(RuntimeError, "model_type"):
                        store._ensure_model_type_metadata(get_model_spec("kokoro"), root)
                    self.assertEqual(config.read_text(), original)

    def test_backend_and_selected_model_checks_share_one_probe(self):
        payload = json.dumps({"available_model_types": ["kokoro"], "probe_error": None})
        with patch("libro_tts.env.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=payload, stderr="")) as run:
            probe = env.validate_mlx_backend_preflight()
            self.assertIs(env.validate_tts_model_runtime_support("kokoro", probe=probe), probe)
            run.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "not supported"):
                env.validate_tts_model_runtime_support("dia", probe=probe)
            run.assert_called_once()

    def test_probe_schema_errors_and_import_failures_are_backend_errors(self):
        for payload in ([], {"available_model_types": None}, {"probe_error": "ImportError: broken"}):
            with self.subTest(payload=payload), patch("libro_tts.env.subprocess.run", return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps(payload), stderr="")), self.assertRaisesRegex(RuntimeError, "MLX backend failed"):
                env.validate_mlx_backend_preflight()
