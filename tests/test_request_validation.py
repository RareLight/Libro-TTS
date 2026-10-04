import io
import logging
from contextlib import ExitStack, redirect_stderr
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from libro_tts.catalog import get_model_spec
from libro_tts.cli import main
from libro_tts.runtime import (
    RuntimeOptions, _normalize_speed, _read_input_text, build_generate_kwargs,
    generate_tts, process_batch_dir, process_single_file,
)
from libro_tts.validation import validate_output_destination, validate_speed


LOGGER = logging.getLogger("test.validation")
OPTIONS = RuntimeOptions("kokoro", None, None, None, None, None, False, True)


class CliRequestValidationTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.input = self.root / "input.txt"
        self.input.write_text("hello")
        self.error = io.StringIO()
        stack.enter_context(redirect_stderr(self.error))
        self.env = stack.enter_context(patch("libro_tts.cli.validate_runtime_environment"))
        self.backend = stack.enter_context(patch("libro_tts.cli.validate_mlx_backend_preflight"))
        stack.enter_context(patch("libro_tts.cli.validate_tts_model_runtime_support"))
        self.encoder = stack.enter_context(patch("libro_tts.cli.validate_local_encoder"))
        self.dirs = stack.enter_context(patch("libro_tts.cli.prepare_runtime_dirs",
                                              return_value={"models_dir": self.root / "models"}))
        self.store = stack.enter_context(patch("libro_tts.store.ModelStore"))
        stack.enter_context(patch("libro_tts.cli.configure_local_cache_environment"))
        stack.enter_context(patch("libro_tts.cli.configure_logging", return_value=LOGGER))

    def assert_no_backend_or_state(self):
        self.backend.assert_not_called()
        self.dirs.assert_not_called()
        self.store.assert_not_called()

    def test_input_modes_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as raised:
            main([str(self.input), "--input-dir", str(self.root)])
        self.assertEqual(raised.exception.code, 2)
        self.assert_no_backend_or_state()

    def test_missing_or_wrong_input_type_fails_before_environment_checks(self):
        for args in ([], [str(self.root / "missing.txt")], [str(self.root)],
                     ["--input-dir", str(self.input)], ["--input-dir", ""]):
            with self.subTest(args=args):
                self.assertEqual(main(args), 1)
        self.env.assert_not_called()
        self.assert_no_backend_or_state()

    def test_invalid_speeds_fail_without_backend_or_state(self):
        for speed in ("nan", "inf", "-inf", "-1", "0", "1e308"):
            with self.subTest(speed=speed):
                self.assertEqual(main([str(self.input), f"--speed={speed}"]), 1)
        self.env.assert_not_called()
        self.assert_no_backend_or_state()

    def test_invalid_audio_format_fails_at_parse_time(self):
        with self.assertRaises(SystemExit) as raised:
            main([str(self.input), "--audio-format", "aac"])
        self.assertEqual(raised.exception.code, 2)
        self.assert_no_backend_or_state()

    def test_invalid_output_fails_before_backend(self):
        output_dir = self.root / "out.wav"
        output_dir.mkdir()
        parent_file = self.root / "parent"
        parent_file.write_text("file")
        for target in (output_dir, parent_file / "out.wav"):
            with self.subTest(target=target):
                self.assertEqual(main([str(self.input), "-o", str(target)]), 1)
        self.assert_no_backend_or_state()

    def test_empty_preprocessed_input_fails_before_backend(self):
        for text in ("", " \n\t", "[1] [ 22 ]", "\ufeff"):
            with self.subTest(text=text):
                self.input.write_text(text)
                self.assertEqual(main([str(self.input)]), 1)
        self.assert_no_backend_or_state()
        self.assertIn("no text", self.error.getvalue())

    def test_cli_rejects_legacy_encoding_and_reports_read_errors(self):
        self.input.write_bytes(b"Caf\xe9")
        self.assertEqual(main([str(self.input)]), 1)
        self.assertIn("Failed to decode", self.error.getvalue())
        with patch.object(Path, "read_text", side_effect=PermissionError("permission denied")):
            self.assertEqual(main([str(self.input)]), 1)
        self.assertIn("permission denied", self.error.getvalue())
        self.assert_no_backend_or_state()

    def test_bom_input_is_read_once_and_reused_by_real_single_dispatch(self):
        self.input.write_text("\ufeffHello caf\u00e9")
        with patch("libro_tts.runtime._read_input_text", wraps=_read_input_text) as read, \
                patch("libro_tts.runtime.generate_tts") as generate:
            self.assertEqual(main([str(self.input)]), 0)
            read.assert_called_once()
            self.assertEqual(generate.call_args.kwargs["text"], "Hello caf\u00e9")
            self.assertEqual(generate.call_args.kwargs["options"].input_encoding_fallbacks, ())
        self.encoder.assert_not_called()
        self.backend.assert_called_once()

    def test_spark_zero_and_case_insensitive_formats_remain_accepted(self):
        for audio_format in ("WAV", "MP3", "pcm", "raw"):
            with self.subTest(audio_format=audio_format), \
                    patch("libro_tts.runtime.process_single_file") as process:
                self.assertEqual(main([str(self.input), "--model", "spark", "--speed", "0",
                                       "--audio-format", audio_format]), 0)
                options = process.call_args.kwargs["options"]
                self.assertEqual(options.speed, 0)
                self.assertEqual(options.audio_format, audio_format)
        self.encoder.assert_called_once()

    def test_encoder_failure_precedes_backend_and_store_creation(self):
        self.encoder.side_effect = RuntimeError("local FFmpeg missing")
        self.assertEqual(main([str(self.input), "--audio-format", "mp3"]), 1)
        self.assertIn("local FFmpeg missing", self.error.getvalue())
        self.assert_no_backend_or_state()


class RuntimeRequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = Mock()
        self.resolve = self.enterContext(patch("libro_tts.runtime._resolve_model_reference"))
        self.load = self.enterContext(patch("libro_tts.runtime._load_runtime_model"))

    def generate(self, text="hello", options=OPTIONS, output=None):
        generate_tts(text=text, output_prefix=str(output or self.root / "output.wav"),
                     model_store=self.store, options=options, logger=LOGGER)

    def assert_no_acquisition(self):
        self.resolve.assert_not_called()
        self.load.assert_not_called()
        self.assertEqual(self.store.mock_calls, [])

    def test_empty_text_and_invalid_options_fail_before_acquisition(self):
        for text in ("", "\n\t ", "[1] [ 22 ]", "\ufeff"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "no text"):
                self.generate(text=text)
        for options in (replace(OPTIONS, speed=float("nan")), replace(OPTIONS, speed=0),
                        replace(OPTIONS, audio_format="aac")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.generate(options=options)
        self.assert_no_acquisition()

    def test_bad_output_and_missing_prompt_fail_before_acquisition(self):
        output_dir = self.root / "output.wav"
        output_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.generate(output=output_dir)
        with self.assertRaisesRegex(FileNotFoundError, "Voice prompt"):
            self.generate(options=replace(OPTIONS, model_key="spark", voice="missing-prompt"),
                          output=self.root / "valid.wav")
        self.assert_no_acquisition()

    def test_direct_generation_checks_encoder_before_acquisition(self):
        with patch("libro_tts.runtime.validate_local_encoder", side_effect=RuntimeError("encoder")):
            with self.assertRaisesRegex(RuntimeError, "encoder"):
                self.generate(options=replace(OPTIONS, audio_format="flac"))
        self.assert_no_acquisition()

    def test_input_alias_cannot_be_overwritten(self):
        source = self.root / "input.txt"
        source.write_text("preserved input")
        alias = self.root / "output.wav"
        alias.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            process_single_file(input_file=source, output_prefix=str(alias), model_store=self.store,
                                options=OPTIONS, logger=LOGGER)
        self.assertEqual(source.read_text(), "preserved input")
        self.assert_no_acquisition()

    def test_input_protection_covers_case_and_unicode_aliases(self):
        for name, alternate in (("input.txt", "INPUT.TXT"), ("caf\u00e9.txt", "cafe\u0301.txt")):
            with self.subTest(name=name):
                source = self.root / name
                source.write_text("preserved input")
                alias = self.root / "output.wav"
                alias.symlink_to(self.root / alternate)
                try:
                    with self.assertRaisesRegex(ValueError, "overwrite an input"):
                        process_single_file(input_file=source, output_prefix=str(alias), model_store=self.store,
                                            options=OPTIONS, logger=LOGGER)
                    self.assertEqual(source.read_text(), "preserved input")
                finally:
                    alias.unlink()
        self.assert_no_acquisition()

    def test_batch_rejects_output_alias_to_another_input_before_mkdir_or_read(self):
        inputs, outputs = self.root / "inputs", self.root / "outputs"
        inputs.mkdir()
        outputs.mkdir()
        (inputs / "first.txt").write_text("first")
        second = inputs / "second.txt"
        second.write_text("second")
        (outputs / "first.wav").symlink_to(second)
        with patch("libro_tts.runtime._read_input_text") as read, \
                self.assertRaisesRegex(ValueError, "overwrite an input"):
            process_batch_dir(input_dir=inputs, output_dir=outputs, model_store=self.store,
                              options=OPTIONS, logger=LOGGER)
        read.assert_not_called()
        self.assertEqual(second.read_text(), "second")
        self.assert_no_acquisition()

    def test_batch_invalid_options_and_encoder_do_not_create_output_directory(self):
        inputs, outputs = self.root / "inputs", self.root / "outputs"
        inputs.mkdir()
        (inputs / "first.txt").write_text("first")
        for options in (replace(OPTIONS, speed=float("inf")), replace(OPTIONS, audio_format="aac")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                process_batch_dir(input_dir=inputs, output_dir=outputs, model_store=self.store,
                                  options=options, logger=LOGGER)
        with patch("libro_tts.runtime.validate_local_encoder", side_effect=RuntimeError("encoder")), \
                self.assertRaisesRegex(RuntimeError, "encoder"):
            process_batch_dir(input_dir=inputs, output_dir=outputs, model_store=self.store,
                              options=replace(OPTIONS, audio_format="mp3"), logger=LOGGER)
        self.assertFalse(outputs.exists())
        self.assert_no_acquisition()

    def test_empty_batch_item_fails_without_acquisition_and_valid_item_continues(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "a_empty.txt").write_text("[1]")
        (inputs / "b_valid.txt").write_text("Hello")
        self.store.root_dir = self.root / "models"
        self.resolve.return_value = (str(self.root / "model"), get_model_spec("soprano"))
        with patch("libro_tts.runtime.validate_model_load_preflight"), \
                patch("libro_tts.runtime.prepare_model_assets"), \
                patch("libro_tts.runtime._generate_audio_serial", return_value=([np.array([0.1])], 24000)), \
                patch("libro_tts.runtime._write_output_audio") as write:
            result = process_batch_dir(input_dir=inputs, output_dir=self.root / "outputs",
                model_store=self.store, options=replace(OPTIONS, model_key="soprano", verbose=True, stream_wav=False), logger=LOGGER)
        self.assertEqual((result.files_processed, result.files_failed), (1, 1))
        self.assertEqual(result.failures[0].input_file, "a_empty.txt")
        self.assertIn("no text", result.failures[0].error)
        self.resolve.assert_called_once()
        self.load.assert_called_once()
        self.assertEqual(Path(write.call_args.kwargs["output_target"]).name, "b_valid.wav")

    def test_nonwritable_parent_is_rejected_without_creating_directories(self):
        target = self.root / "new" / "output.wav"
        with patch("libro_tts.validation.os.access", return_value=False), \
                self.assertRaisesRegex(ValueError, "not writable"):
            validate_output_destination(target)
        self.assertFalse(target.parent.exists())

    def test_library_bom_and_legacy_fallback_contract_is_preserved(self):
        source = self.root / "input.txt"
        for data in (b"\xef\xbb\xbfCaf\xc3\xa9", b"Caf\xe9"):
            with self.subTest(data=data):
                source.write_bytes(data)
                with redirect_stderr(io.StringIO()):
                    text = _read_input_text(input_file=source, options=OPTIONS, logger=LOGGER)
                self.assertEqual(text, "Caf\u00e9")

    def test_model_speed_normalization_preserves_spark_categories_and_override(self):
        for speed, expected in ((0, 0), (0.2, 0), (0.6, 0.5), (1.1, 1), (1.6, 1.5), (3, 2)):
            with self.subTest(speed=speed):
                self.assertEqual(_normalize_speed("spark", speed), expected)
                kwargs = build_generate_kwargs(text="Hello", spec=None, model_key="spark",
                    voice=None, speed=speed, lang_code=None, max_tokens=None, verbose=False)
                self.assertEqual(kwargs["speed"], expected)
        for key in ("kokoro", "spark", None):
            for speed in (float("nan"), float("inf"), -1):
                with self.subTest(key=key, speed=speed), self.assertRaises(ValueError):
                    validate_speed(key, speed)
        self.assertEqual(_normalize_speed("kokoro", 1.1), 1.1)
