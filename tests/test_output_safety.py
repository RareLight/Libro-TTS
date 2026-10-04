import io
import logging
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from libro_tts.cli import main
from libro_tts.runtime import (
    RuntimeOptions, _plan_batch_outputs, _write_output_audio, process_batch_dir,
)
import libro_tts.runtime as runtime
from libro_tts.store import ModelStore


class BatchCollisionTests(unittest.TestCase):
    def test_normalized_name_collisions_are_rejected_for_all_output_formats(self):
        examples = [
            ("chapter.txt", "chapter.wav.txt"),
            ("chapter.txt", "chapter.MP3.txt"),
            ("chapter.txt", ".chapter.txt"),
            (".txt", "txt.txt"),
            (".wav.txt", "untitled.txt"),
            ("chapter.1.txt", "chapter.1.flac.txt"),
            ("Chapter.txt", "chapter.txt"),
            ("caf\u00e9.txt", "cafe\u0301.txt"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for names in examples:
                for audio_format in ("wav", "mp3", "flac"):
                    with self.subTest(names=names, format=audio_format):
                        # Synthetic paths let macOS test distinct case/Unicode
                        # spellings even when its volume cannot store both.
                        files = [Path(temporary) / name for name in names]
                        with self.assertRaisesRegex(ValueError, "collisions") as raised:
                            _plan_batch_outputs(files, Path(temporary) / "outputs", audio_format)
                        for name in names:
                            self.assertIn(repr(name), str(raised.exception))

    def test_unique_dotted_and_hidden_names_keep_existing_destinations(self):
        names = ["chapter.1.txt", "chapter.2.txt", "foo.wav.txt", ".txt", ".hidden.txt"]
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "outputs"
            files = [Path(temporary) / name for name in names]
            plan = _plan_batch_outputs(files, output_dir, "mp3")
            self.assertEqual([source for source, _ in plan], files)
            self.assertEqual([target.name for _, target in plan],
                             ["chapter.1.mp3", "chapter.2.mp3", "foo.mp3", "txt.mp3", "hidden.mp3"])
            self.assertFalse(output_dir.exists())

    def test_batch_collision_stops_every_file_before_reading_or_acquisition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            for name in ("a_unique.txt", "chapter.txt", "chapter.wav.txt"):
                (inputs / name).write_text("text")
            outputs = root / "outputs"
            store = Mock()
            options = RuntimeOptions("kokoro", None, None, None, None, None, False, True)
            with patch("libro_tts.runtime.process_single_file") as process, \
                    patch("libro_tts.runtime._read_input_text") as read, \
                    patch("libro_tts.runtime._get_load_model_fn") as load:
                with self.assertRaisesRegex(ValueError, "no files were processed"):
                    process_batch_dir(input_dir=inputs, output_dir=outputs, model_store=store,
                                      options=options, logger=logging.getLogger("test"))
                process.assert_not_called()
                read.assert_not_called()
                load.assert_not_called()
                self.assertEqual(store.mock_calls, [])
            self.assertFalse(outputs.exists())

    def test_collision_preserves_existing_outputs_and_reports_all_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, outputs = root / "inputs", root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            for name in ("chapter.txt", "chapter.wav.txt", "other.txt", ".other.txt"):
                (inputs / name).write_text("text")
            (outputs / "chapter.wav").write_bytes(b"old complete audio")
            options = RuntimeOptions("kokoro", None, None, None, None, None, False, True)
            with self.assertRaises(ValueError) as raised:
                process_batch_dir(input_dir=inputs, output_dir=outputs, model_store=Mock(),
                                  options=options, logger=logging.getLogger("test"))
            self.assertIn("chapter.wav", str(raised.exception))
            self.assertIn("other.wav", str(raised.exception))
            self.assertEqual((outputs / "chapter.wav").read_bytes(), b"old complete audio")
            self.assertEqual([path.name for path in outputs.iterdir()], ["chapter.wav"])

    def test_existing_output_symlinks_to_same_destination_collide(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "shared.wav"
            target.write_bytes(b"complete audio")
            outputs = root / "outputs"
            outputs.mkdir()
            (outputs / "one.wav").symlink_to(target)
            (outputs / "two.wav").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "collisions"):
                _plan_batch_outputs([root / "one.txt", root / "two.txt"], outputs, "wav")
            self.assertEqual(target.read_bytes(), b"complete audio")

    def test_cli_returns_nonzero_for_real_collision_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "chapter.txt").write_text("text")
            (inputs / "chapter.wav.txt").write_text("text")
            error = io.StringIO()
            with patch("libro_tts.cli.validate_runtime_environment"), \
                    patch("libro_tts.cli.validate_mlx_backend_preflight"), \
                    patch("libro_tts.cli.validate_tts_model_runtime_support"), \
                    patch("libro_tts.cli.prepare_runtime_dirs", return_value={"models_dir": root / "models"}), \
                    patch("libro_tts.store.ModelStore") as store, \
                    patch("libro_tts.runtime._get_load_model_fn") as load, \
                    patch("sys.stderr", error):
                self.assertEqual(main(["--input-dir", str(inputs), "-o", str(root / "outputs")]), 1)
                store.return_value.ensure_model.assert_not_called()
                load.assert_not_called()
            self.assertIn("Batch output collisions", error.getvalue())
            self.assertIn("chapter.wav.txt", error.getvalue())
            self.assertFalse((root / "outputs").exists())

    def test_batch_continues_after_encoding_failure_and_preserves_previous_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, outputs, model_dir = root / "inputs", root / "outputs", root / "model"
            inputs.mkdir()
            outputs.mkdir()
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type":"soprano"}')
            (model_dir / "model.safetensors").write_text("fixture weights")
            for name in ("bad", "good"):
                (inputs / f"{name}.txt").write_text(name)
            (outputs / "bad.mp3").write_bytes(b"old complete audio")
            def generate(**kwargs):
                yield SimpleNamespace(audio=np.array([0.2 if kwargs["text"] == "bad" else 0.1]), sample_rate=24000)
            model = SimpleNamespace(generate=generate, sample_rate=24000)
            def encoder(file, audio, *_args, **_kwargs):
                if audio[0] > 0.15:
                    Path(file).write_bytes(b"partial")
                    raise RuntimeError("encoding failed")
                Path(file).write_bytes(b"complete good audio")
            options = RuntimeOptions("soprano", str(model_dir), None, None, None, None, False, True, audio_format="mp3")
            runtime._LOADED_MODEL_CACHE.clear()
            try:
                with patch("libro_tts.runtime._get_load_model_fn", return_value=lambda _path: model), \
                        patch("libro_tts.runtime._get_audio_write_fn", return_value=encoder), \
                        patch("libro_tts.runtime._print_user_message"):
                    result = process_batch_dir(input_dir=inputs, output_dir=outputs,
                        model_store=ModelStore(root / "models"), options=options, logger=logging.getLogger("test"))
                self.assertEqual((result.total_files, result.files_processed, result.files_failed), (2, 1, 1))
                self.assertEqual(result.failures[0].input_file, "bad.txt")
                self.assertIn("encoding failed", result.failures[0].error)
                self.assertEqual((outputs / "bad.mp3").read_bytes(), b"old complete audio")
                self.assertEqual((outputs / "good.mp3").read_bytes(), b"complete good audio")
                self.assertFalse(list(outputs.glob(".libro-audio-*")))
            finally:
                runtime._LOADED_MODEL_CACHE.clear()


class AtomicOutputTests(unittest.TestCase):
    def write(self, target, audio_format="wav"):
        return _write_output_audio(output_target=str(target), audio_format=audio_format,
                                   audio=np.array([0.1, -0.1], dtype=np.float32), sample_rate=24000)

    def test_success_publishes_only_complete_audio_and_preserves_existing_mode(self):
        for audio_format in ("wav", "mp3", "flac"):
            with self.subTest(format=audio_format), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / f"chapter.{audio_format}"
                target.write_bytes(b"old complete audio")
                target.chmod(0o640)
                seen = []
                def encoder(file, audio, sample_rate, format):
                    staged = Path(file)
                    self.assertEqual(staged.parent, target.parent.resolve())
                    self.assertEqual(staged.suffix, target.suffix)
                    self.assertNotEqual(staged, target.resolve())
                    self.assertEqual(target.read_bytes(), b"old complete audio")
                    staged.write_bytes(b"partial")
                    self.assertEqual(target.read_bytes(), b"old complete audio")
                    staged.write_bytes(b"new complete audio")
                    seen.append((len(audio), sample_rate, format))
                with patch("libro_tts.runtime._get_audio_write_fn", return_value=encoder):
                    self.assertEqual(self.write(target, audio_format), target)
                self.assertEqual(target.read_bytes(), b"new complete audio")
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
                self.assertEqual(seen, [(2, 24000, audio_format)])
                self.assertEqual(list(target.parent.iterdir()), [target])

    def test_encoder_failure_and_interruption_preserve_old_output_and_clean_staging(self):
        for failure in (RuntimeError("ffmpeg failed: encoder error"), OSError("disk full"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "chapter.mp3"
                target.write_bytes(b"old complete audio")
                def encoder(file, *_args, **_kwargs):
                    Path(file).write_bytes(b"partial encoded audio")
                    raise failure
                with patch("libro_tts.runtime._get_audio_write_fn", return_value=encoder):
                    with self.assertRaises(KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else RuntimeError):
                        self.write(target, "mp3")
                self.assertEqual(target.read_bytes(), b"old complete audio")
                self.assertEqual(list(target.parent.iterdir()), [target])

    def test_failed_new_output_never_publishes_partial_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new directory/chapter.wav"
            def encoder(file, *_args, **_kwargs):
                Path(file).write_bytes(b"partial")
                raise RuntimeError("encoder failure")
            with patch("libro_tts.runtime._get_audio_write_fn", return_value=encoder):
                with self.assertRaisesRegex(RuntimeError, "encoder failure"):
                    self.write(target)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])

    def test_silent_empty_encoder_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "chapter.wav"
            target.write_bytes(b"old complete audio")
            with patch("libro_tts.runtime._get_audio_write_fn", return_value=Mock()):
                with self.assertRaisesRegex(RuntimeError, "empty output"):
                    self.write(target)
            self.assertEqual(target.read_bytes(), b"old complete audio")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_sync_and_publication_failures_preserve_existing_output(self):
        for failing_call in ("os.fsync", "os.replace"):
            with self.subTest(call=failing_call), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "chapter.wav"
                target.write_bytes(b"old complete audio")
                def encoder(file, *_args, **_kwargs):
                    Path(file).write_bytes(b"new complete audio")
                with patch("libro_tts.runtime._get_audio_write_fn", return_value=encoder), \
                        patch("libro_tts.audio_output." + failing_call, side_effect=OSError("publication failure")):
                    with self.assertRaisesRegex(RuntimeError, "Failed to write audio output"):
                        self.write(target)
                self.assertEqual(target.read_bytes(), b"old complete audio")
                self.assertEqual(list(target.parent.iterdir()), [target])

    def test_successful_new_file_keeps_normal_creation_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ordinary = root / "ordinary"
            ordinary.write_bytes(b"file")
            target = root / "chapter.wav"
            def encoder(file, *_args, **_kwargs):
                Path(file).write_bytes(b"complete audio")
            with patch("libro_tts.runtime._get_audio_write_fn", return_value=encoder):
                self.write(target)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), stat.S_IMODE(ordinary.stat().st_mode))
            self.assertFalse(list(root.glob(".libro-audio-*")))

    def test_output_symlink_is_preserved_and_its_target_is_atomically_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "actual/chapter.wav"
            destination.parent.mkdir()
            destination.write_bytes(b"old complete audio")
            link = root / "requested.wav"
            link.symlink_to(destination)
            def encoder(file, *_args, **_kwargs):
                self.assertEqual(Path(file).parent, destination.parent.resolve())
                self.assertEqual(destination.read_bytes(), b"old complete audio")
                Path(file).write_bytes(b"new complete audio")
            with patch("libro_tts.runtime._get_audio_write_fn", return_value=encoder):
                self.assertEqual(self.write(link), link)
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.read_bytes(), b"new complete audio")
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_temporary_creation_collision_does_not_remove_someone_elses_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "chapter.wav"
            target.write_bytes(b"old complete audio")
            foreign = root / (".libro-audio-" + "a" * 32 + ".tmp.wav")
            foreign.write_bytes(b"another writer's temporary file")
            with patch("libro_tts.audio_output.uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
                    patch("libro_tts.runtime._get_audio_write_fn") as encoder:
                with self.assertRaisesRegex(RuntimeError, "Failed to write audio output"):
                    self.write(target)
                encoder.assert_not_called()
            self.assertEqual(foreign.read_bytes(), b"another writer's temporary file")
            self.assertEqual(target.read_bytes(), b"old complete audio")


if __name__ == "__main__":
    unittest.main()
