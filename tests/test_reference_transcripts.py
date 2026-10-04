import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from libro_tts.catalog import get_model_spec
import libro_tts.runtime as runtime
from libro_tts.runtime import (
    RuntimeOptions, SPARK_MAX_REF_TEXT_CHARS, _auto_transcribe_reference_text,
    _populate_missing_ref_text, _resolve_reference_text, generate_tts,
)


class ReferenceTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.audio = self.root / "speaker.wav"
        self.audio.write_bytes(b"reference fixture")
        self.transcript = self.audio.with_suffix(".txt")
        self.logger = Mock(spec=logging.Logger)

    def populate(self, kwargs=None, model="dia"):
        kwargs = kwargs if kwargs is not None else {"ref_audio": str(self.audio)}
        return _populate_missing_ref_text(model_key=model, kwargs=kwargs,
                                          logger=self.logger, verbose=False)

    def test_missing_empty_whitespace_and_bom_only_files_use_stt(self):
        for data in (None, b"", b" \n\t", b"\xef\xbb\xbf", " \ufeff \n".encode()):
            with self.subTest(data=data):
                if data is not None:
                    self.transcript.write_bytes(data)
                kwargs = {"ref_audio": str(self.audio)}
                with patch("libro_tts.runtime._auto_transcribe_reference_text", return_value="\ufeff spoken\n words ") as stt:
                    resolved = self.populate(kwargs)
                stt.assert_called_once()
                self.assertEqual((resolved.text, resolved.source), ("spoken words", "stt"))
                self.assertEqual(kwargs, {"ref_audio": str(self.audio), "ref_text": "spoken words"})

    def test_file_encoding_normalization_and_provenance(self):
        for data in ("\ufeffCaf\u00e9\n\twords".encode(), b"Caf\xe9\n\twords"):
            with self.subTest(data=data):
                self.transcript.write_bytes(data)
                with patch("libro_tts.runtime._auto_transcribe_reference_text") as stt:
                    resolved = self.populate()
                self.assertEqual((resolved.text, resolved.source), ("Caf\u00e9 words", "file"))
                stt.assert_not_called()

    def test_explicit_text_wins_without_reading_file_or_transcribing(self):
        kwargs = {"ref_audio": str(self.audio), "ref_text": " \ufeff explicit\n words "}
        with patch("libro_tts.runtime._resolve_reference_text") as read, \
                patch("libro_tts.runtime._auto_transcribe_reference_text") as stt:
            resolved = self.populate(kwargs)
        self.assertEqual((resolved.text, resolved.source), ("explicit words", "explicit"))
        read.assert_not_called()
        stt.assert_not_called()
        self.assertEqual(set(kwargs), {"ref_audio", "ref_text"})

    def test_empty_explicit_text_uses_sibling_file(self):
        self.transcript.write_text("from file")
        for text in ("", " \t\n", "\ufeff"):
            with self.subTest(text=text):
                resolved = self.populate({"ref_audio": str(self.audio), "ref_text": text})
                self.assertEqual((resolved.text, resolved.source), ("from file", "file"))

    def test_transcript_read_failure_is_not_silently_replaced_by_stt(self):
        self.transcript.write_text("from file")
        with patch.object(Path, "read_text", side_effect=PermissionError("unreadable")), \
                patch("libro_tts.runtime._auto_transcribe_reference_text") as stt, \
                self.assertRaisesRegex(PermissionError, "unreadable"):
            self.populate()
        stt.assert_not_called()

    def test_required_transcription_failure_and_empty_result_are_errors(self):
        with patch("libro_tts.runtime._auto_transcribe_reference_text", side_effect=RuntimeError("STT failure")), \
                self.assertRaisesRegex(RuntimeError, "STT failure"):
            self.populate()
        with patch("libro_tts.runtime._auto_transcribe_reference_text", return_value=" \n\ufeff "), \
                self.assertRaisesRegex(RuntimeError, "empty"):
            self.populate()

    def test_long_spark_file_keeps_trimmed_original_when_transcription_fails_or_is_empty(self):
        text = "Original reference sentence. " * 50
        self.transcript.write_text(text)
        for result in (RuntimeError("offline asset absent"), " \n\ufeff ", None):
            stt = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
            kwargs = {"ref_audio": str(self.audio)}
            with self.subTest(result=result), patch("libro_tts.runtime._auto_transcribe_reference_text", stt):
                resolved = self.populate(kwargs, model="spark")
            self.assertEqual(resolved.source, "file")
            self.assertTrue(resolved.text.startswith("Original reference sentence."))
            self.assertLessEqual(len(resolved.text), SPARK_MAX_REF_TEXT_CHARS)
            self.assertTrue(resolved.text.endswith("."))
            self.assertEqual(kwargs["ref_text"], resolved.text)
            stt.assert_called_once()

    def test_long_spark_stt_is_capped_and_retains_stt_provenance(self):
        self.transcript.write_text("Original reference sentence. " * 50)
        with patch("libro_tts.runtime._auto_transcribe_reference_text", return_value="Aligned speech. " * 50):
            resolved = self.populate(model="spark")
        self.assertEqual(resolved.source, "stt")
        self.assertTrue(resolved.text.startswith("Aligned speech."))
        self.assertLessEqual(len(resolved.text), SPARK_MAX_REF_TEXT_CHARS)

    def test_long_explicit_spark_text_is_trimmed_without_stt(self):
        with patch("libro_tts.runtime._auto_transcribe_reference_text") as stt:
            resolved = self.populate({"ref_audio": str(self.audio), "ref_text": "Explicit words. " * 50}, model="spark")
        self.assertEqual(resolved.source, "explicit")
        self.assertLessEqual(len(resolved.text), SPARK_MAX_REF_TEXT_CHARS)
        stt.assert_not_called()

    def test_stt_result_contract_rejects_none_and_empty_text_without_real_models(self):
        store = SimpleNamespace(root_dir=self.root)
        results = (None, SimpleNamespace(text=None), SimpleNamespace(text=" \n\ufeff "),
                   SimpleNamespace(other="text"), "\ufeff words\n here ", SimpleNamespace(text="words here"))
        for result in results:
            with self.subTest(result=result), \
                    patch.object(runtime, "_LOADED_STT_MODEL", None), \
                    patch.object(runtime, "_LOADED_STT_REFERENCE", None), \
                    patch("libro_tts.runtime.ensure_asset", return_value=self.root / "whisper") as ensure, \
                    patch("libro_tts.runtime._get_stt_load_fn", return_value=Mock(return_value=SimpleNamespace(
                        generate=Mock(return_value=result)))):
                kwargs = dict(ref_audio_path=str(self.audio), logger=self.logger, verbose=False,
                              model_store=store, offline=True)
                if result is results[-1] or isinstance(result, str):
                    self.assertEqual(_auto_transcribe_reference_text(**kwargs), "words here")
                else:
                    with self.assertRaisesRegex(RuntimeError, "empty or invalid"):
                        _auto_transcribe_reference_text(**kwargs)
                ensure.assert_called_once_with(store, runtime.WHISPER, True)

    def test_normal_pipeline_resolves_file_once_and_shares_text_across_chunks(self):
        self.transcript.write_text("\ufeffReference\n words")
        model = SimpleNamespace(sample_rate=24000, generate=Mock(return_value=[
            SimpleNamespace(audio=np.array([0.1, 0.2], dtype=np.float32), sample_rate=24000)]))
        options = RuntimeOptions("csm", None, str(self.audio), None, None, None, True, True,
                                 parallel_workers=1)
        with patch("libro_tts.runtime._resolve_model_reference", return_value=(str(self.root), get_model_spec("csm"))), \
                patch("libro_tts.runtime.validate_model_load_preflight"), \
                patch("libro_tts.runtime.prepare_model_assets"), \
                patch("libro_tts.runtime._load_runtime_model", return_value=model), \
                patch("libro_tts.runtime._write_output_audio"), \
                patch("libro_tts.runtime._resolve_reference_text", wraps=_resolve_reference_text) as read, \
                patch("libro_tts.runtime._auto_transcribe_reference_text") as stt:
            generate_tts(text="A complete sentence. " * 50, output_prefix=str(self.root / "output.wav"),
                         model_store=SimpleNamespace(root_dir=self.root), options=options, logger=self.logger)
        read.assert_called_once_with(str(self.audio))
        stt.assert_not_called()
        self.assertGreater(model.generate.call_count, 1)
        for call in model.generate.call_args_list:
            self.assertEqual(call.kwargs["ref_text"], "Reference words")
            self.assertNotIn("source", call.kwargs)
        self.logger.info.assert_any_call("Using %s reference transcript for '%s'", "file", str(self.audio))
