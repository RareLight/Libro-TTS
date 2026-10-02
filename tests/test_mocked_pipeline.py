import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from libro_tts.catalog import REQUIRED_MODEL_KEYS, canonical_model_type_for_key
from libro_tts.paths import project_root
import libro_tts.runtime as runtime
from libro_tts.runtime import RuntimeOptions, process_batch_dir, process_single_file
from libro_tts.store import ModelStore


class MockedPipelineTests(unittest.TestCase):
    def test_processes_input_and_public_plaintext_sources(self):
        root = project_root()
        inputs = [
            root / "input.txt",
            root / "tests" / "fixtures" / "public_domain" / "alice_excerpt.txt",
            root / "tests" / "fixtures" / "public_domain" / "federalist_1_excerpt.txt",
        ]

        for path in inputs:
            self.assertTrue(path.exists(), f"Missing test input: {path}")

        class FakeResult:
            def __init__(self, audio, sample_rate=24000):
                self.audio = audio
                self.sample_rate = sample_rate

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                _ = kwargs
                yield FakeResult(np.array([0.1, 0.2], dtype=np.float32))
                yield FakeResult(np.array([0.3], dtype=np.float32))

        def fake_load_model(_model_reference: str):
            return FakeModel()

        write_calls = []

        def fake_audio_write(file, data, samplerate, format=None):
            output_file = Path(file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("fake audio", encoding="utf-8")
            write_calls.append((output_file, np.asarray(data).shape[0], samplerate, format))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            fake_model_dir = tmp / "fake_model"
            fake_model_dir.mkdir(parents=True, exist_ok=True)
            (fake_model_dir / "config.json").write_text("{}", encoding="utf-8")
            (fake_model_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            output_dir = tmp / "outputs"
            model_store = ModelStore(root_dir=tmp / "models")
            options = RuntimeOptions(
                model_key="kokoro",
                model_path_override=str(fake_model_dir),
                voice=None,
                speed=None,
                lang_code=None,
                max_tokens=None,
                verbose=False,
                offline=True,
                audio_format="wav",
                parallel_workers=1,
            )

            runtime._LOADED_MODEL_CACHE.clear()
            with patch("libro_tts.runtime._get_load_model_fn", return_value=fake_load_model), patch(
                "libro_tts.runtime._get_audio_write_fn", return_value=fake_audio_write
            ), patch("libro_tts.runtime._print_user_message"):
                for source in inputs:
                    out_prefix = str(output_dir / source.stem)
                    result = process_single_file(
                        input_file=source,
                        output_prefix=out_prefix,
                        model_store=model_store,
                        options=options,
                        logger=__import__("logging").getLogger("test"),
                    )
                    self.assertEqual(result.files_processed, 1)
                    self.assertTrue(Path(f"{out_prefix}.wav").exists())

            self.assertEqual(len(write_calls), len(inputs))
            for _path, sample_count, sample_rate, audio_format in write_calls:
                self.assertGreaterEqual(sample_count, 3)
                self.assertEqual(sample_count % 3, 0)
                self.assertEqual(sample_rate, 24000)
                self.assertEqual(audio_format, "wav")

    def test_single_file_decodes_with_fallback_encoding(self):
        class FakeResult:
            def __init__(self, audio, sample_rate=24000):
                self.audio = audio
                self.sample_rate = sample_rate

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                _ = kwargs
                yield FakeResult(np.array([0.1, 0.2], dtype=np.float32))

        def fake_load_model(_model_reference: str):
            return FakeModel()

        def fake_audio_write(file, data, samplerate, format=None):
            _ = data, samplerate, format
            output_file = Path(file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("fake audio", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            fake_model_dir = tmp / "fake_model"
            fake_model_dir.mkdir(parents=True, exist_ok=True)
            (fake_model_dir / "config.json").write_text("{}", encoding="utf-8")
            (fake_model_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            cp1252_input = tmp / "legacy.txt"
            cp1252_input.write_bytes("Caf\xe9 chapter text".encode("latin-1"))

            model_store = ModelStore(root_dir=tmp / "models")
            options = RuntimeOptions(
                model_key="kokoro",
                model_path_override=str(fake_model_dir),
                voice=None,
                speed=None,
                lang_code=None,
                max_tokens=None,
                verbose=False,
                offline=True,
                audio_format="wav",
                parallel_workers=1,
            )

            runtime._LOADED_MODEL_CACHE.clear()
            with patch("libro_tts.runtime._get_load_model_fn", return_value=fake_load_model), patch(
                "libro_tts.runtime._get_audio_write_fn", return_value=fake_audio_write
            ), patch("libro_tts.runtime._print_user_message"):
                result = process_single_file(
                    input_file=cp1252_input,
                    output_prefix=str(tmp / "legacy_out"),
                    model_store=model_store,
                    options=options,
                    logger=__import__("logging").getLogger("test"),
                )

            self.assertEqual(result.files_processed, 1)
            self.assertTrue((tmp / "legacy_out.wav").exists())

    def test_single_file_output_path_with_extension_is_not_duplicated(self):
        class FakeResult:
            def __init__(self, audio, sample_rate=24000):
                self.audio = audio
                self.sample_rate = sample_rate

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                _ = kwargs
                yield FakeResult(np.array([0.1], dtype=np.float32))

        def fake_load_model(_model_reference: str):
            return FakeModel()

        written_paths: list[Path] = []

        def fake_audio_write(file, data, samplerate, format=None):
            _ = data, samplerate, format
            output_file = Path(file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("fake audio", encoding="utf-8")
            written_paths.append(output_file)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            fake_model_dir = tmp / "fake_model"
            fake_model_dir.mkdir(parents=True, exist_ok=True)
            (fake_model_dir / "config.json").write_text("{}", encoding="utf-8")
            (fake_model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
            input_file = tmp / "input.txt"
            input_file.write_text("hello", encoding="utf-8")

            model_store = ModelStore(root_dir=tmp / "models")
            options = RuntimeOptions(
                model_key="kokoro",
                model_path_override=str(fake_model_dir),
                voice=None,
                speed=None,
                lang_code=None,
                max_tokens=None,
                verbose=False,
                offline=True,
                audio_format="wav",
                parallel_workers=1,
            )

            runtime._LOADED_MODEL_CACHE.clear()
            with patch("libro_tts.runtime._get_load_model_fn", return_value=fake_load_model), patch(
                "libro_tts.runtime._get_audio_write_fn", return_value=fake_audio_write
            ), patch("libro_tts.runtime._print_user_message"):
                process_single_file(
                    input_file=input_file,
                    output_prefix=str(tmp / "chapter.wav"),
                    model_store=model_store,
                    options=options,
                    logger=__import__("logging").getLogger("test"),
                )

            self.assertEqual(len(written_paths), 1)
            self.assertEqual(written_paths[0].name, "chapter.wav")

    def test_batch_discovers_case_insensitive_txt_and_reports_failures(self):
        class FakeResult:
            def __init__(self, audio, sample_rate=24000):
                self.audio = audio
                self.sample_rate = sample_rate

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                if "FAIL_ME" in kwargs.get("text", ""):
                    yield None
                    return
                yield FakeResult(np.array([0.1], dtype=np.float32))

        def fake_load_model(_model_reference: str):
            return FakeModel()

        written_paths: list[Path] = []

        def fake_audio_write(file, data, samplerate, format=None):
            _ = data, samplerate, format
            output_file = Path(file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("fake audio", encoding="utf-8")
            written_paths.append(output_file)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            fake_model_dir = tmp / "fake_model"
            fake_model_dir.mkdir(parents=True, exist_ok=True)
            (fake_model_dir / "config.json").write_text("{}", encoding="utf-8")
            (fake_model_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            input_dir = tmp / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "06 - Chapter 1. When Was Canada_.TXT").write_text(
                "hello",
                encoding="utf-8",
            )
            (input_dir / "bad.txt").write_text("FAIL_ME", encoding="utf-8")
            (input_dir / "notes.md").write_text("ignore me", encoding="utf-8")

            model_store = ModelStore(root_dir=tmp / "models")
            options = RuntimeOptions(
                model_key="kokoro",
                model_path_override=str(fake_model_dir),
                voice=None,
                speed=None,
                lang_code=None,
                max_tokens=None,
                verbose=False,
                offline=True,
                audio_format="wav",
                parallel_workers=1,
            )

            runtime._LOADED_MODEL_CACHE.clear()
            with patch("libro_tts.runtime._get_load_model_fn", return_value=fake_load_model), patch(
                "libro_tts.runtime._get_audio_write_fn", return_value=fake_audio_write
            ), patch("libro_tts.runtime._print_user_message"):
                result = process_batch_dir(
                    input_dir=input_dir,
                    output_dir=tmp / "outs",
                    model_store=model_store,
                    options=options,
                    logger=__import__("logging").getLogger("test"),
                )

            self.assertEqual(result.total_files, 2)
            self.assertEqual(result.files_processed, 1)
            self.assertEqual(result.files_failed, 1)
            self.assertEqual(result.failures[0].input_file, "bad.txt")
            self.assertEqual(len(written_paths), 1)
            self.assertTrue(written_paths[0].name.endswith(".wav"))

    def test_batch_output_avoids_double_extension_and_hidden_names(self):
        class FakeResult:
            def __init__(self, audio, sample_rate=24000):
                self.audio = audio
                self.sample_rate = sample_rate

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                _ = kwargs
                yield FakeResult(np.array([0.1], dtype=np.float32))

        def fake_load_model(_model_reference: str):
            return FakeModel()

        written_paths: list[Path] = []

        def fake_audio_write(file, data, samplerate, format=None):
            _ = data, samplerate, format
            output_file = Path(file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("fake audio", encoding="utf-8")
            written_paths.append(output_file)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            fake_model_dir = tmp / "fake_model"
            fake_model_dir.mkdir(parents=True, exist_ok=True)
            (fake_model_dir / "config.json").write_text("{}", encoding="utf-8")
            (fake_model_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            input_dir = tmp / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "foo.wav.txt").write_text("hello", encoding="utf-8")
            (input_dir / ".txt").write_text("hello", encoding="utf-8")

            model_store = ModelStore(root_dir=tmp / "models")
            options = RuntimeOptions(
                model_key="kokoro",
                model_path_override=str(fake_model_dir),
                voice=None,
                speed=None,
                lang_code=None,
                max_tokens=None,
                verbose=False,
                offline=True,
                audio_format="wav",
                parallel_workers=1,
            )

            runtime._LOADED_MODEL_CACHE.clear()
            with patch("libro_tts.runtime._get_load_model_fn", return_value=fake_load_model), patch(
                "libro_tts.runtime._get_audio_write_fn", return_value=fake_audio_write
            ), patch("libro_tts.runtime._print_user_message"):
                result = process_batch_dir(
                    input_dir=input_dir,
                    output_dir=tmp / "outs",
                    model_store=model_store,
                    options=options,
                    logger=__import__("logging").getLogger("test"),
                )

            self.assertEqual(result.files_processed, 2)
            names = sorted(path.name for path in written_paths)
            self.assertEqual(names, ["foo.wav", "txt.wav"])

    def test_all_models_process_single_file_with_expected_kwargs(self):
        class FakeResult:
            def __init__(self, audio, sample_rate=24000):
                self.audio = audio
                self.sample_rate = sample_rate

        captured_calls: list[tuple[str, dict]] = []
        model_by_reference: dict[str, str] = {}

        class FakeModel:
            sample_rate = 24000

            def __init__(self, model_key: str):
                self.model_key = model_key

            def generate(self, **kwargs):
                captured_calls.append((self.model_key, dict(kwargs)))
                yield FakeResult(np.array([0.1, 0.2], dtype=np.float32))

        def fake_load_model(model_reference: str):
            model_key = model_by_reference[str(Path(model_reference).resolve())]
            return FakeModel(model_key)

        def fake_audio_write(file, data, samplerate, format=None):
            _ = data, samplerate, format
            output_file = Path(file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("fake audio", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_file = tmp / "input.txt"
            input_file.write_text("hello from emma prompt test", encoding="utf-8")

            model_store = ModelStore(root_dir=tmp / "models")
            ref_audio_path = (project_root() / "reference_voices" / "Emma.wav").resolve()
            ref_text_value = (project_root() / "reference_voices" / "Emma.txt").read_text(
                encoding="utf-8"
            ).strip()

            runtime._LOADED_MODEL_CACHE.clear()
            with patch("libro_tts.runtime._get_load_model_fn", return_value=fake_load_model), patch(
                "libro_tts.runtime._get_audio_write_fn", return_value=fake_audio_write
            ), patch("libro_tts.runtime._print_user_message"), patch(
                "libro_tts.runtime.validate_model_load_preflight"
            ):
                for model_key in REQUIRED_MODEL_KEYS:
                    model_dir = tmp / "models_local" / model_key
                    model_dir.mkdir(parents=True, exist_ok=True)
                    (model_dir / "config.json").write_text(
                        f'{{"model_type":"{canonical_model_type_for_key(model_key)}"}}',
                        encoding="utf-8",
                    )
                    (model_dir / "weights.safetensors").write_text("weights", encoding="utf-8")
                    model_by_reference[str(model_dir.resolve())] = model_key

                    options = RuntimeOptions(
                        model_key=model_key,
                        model_path_override=str(model_dir),
                        voice="Emma" if model_key in runtime.VOICE_PROMPT_MODELS else None,
                        speed=None,
                        lang_code=None,
                        max_tokens=None,
                        verbose=False,
                        offline=True,
                        audio_format="wav",
                        parallel_workers=1,
                    )

                    result = process_single_file(
                        input_file=input_file,
                        output_prefix=str(tmp / "outs" / model_key),
                        model_store=model_store,
                        options=options,
                        logger=__import__("logging").getLogger("test"),
                    )
                    self.assertEqual(result.files_processed, 1)
                    self.assertTrue((tmp / "outs" / f"{model_key}.wav").exists())

        self.assertEqual(len(captured_calls), len(REQUIRED_MODEL_KEYS))
        kwargs_by_model = {model_key: kwargs for model_key, kwargs in captured_calls}

        for model_key in REQUIRED_MODEL_KEYS:
            kwargs = kwargs_by_model[model_key]
            if model_key in runtime.VOICE_PROMPT_MODELS:
                self.assertEqual(kwargs.get("ref_audio"), str(ref_audio_path))
            else:
                self.assertNotIn("ref_audio", kwargs)

            if model_key in runtime.REF_TEXT_PROMPT_MODELS:
                self.assertEqual(kwargs.get("ref_text"), ref_text_value)
            else:
                self.assertNotIn("ref_text", kwargs)

        self.assertNotIn("voice", kwargs_by_model["qwen3_tts"])

    def test_voxtral_named_voice_override_is_passed_through_generation(self):
        class FakeResult:
            def __init__(self, audio, sample_rate=24000):
                self.audio = audio
                self.sample_rate = sample_rate

        captured_kwargs: dict[str, object] = {}

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                captured_kwargs.update(kwargs)
                yield FakeResult(np.array([0.1, 0.2], dtype=np.float32))

        def fake_load_model(_model_reference: str):
            return FakeModel()

        def fake_audio_write(file, data, samplerate, format=None):
            _ = data, samplerate, format
            output_file = Path(file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("fake audio", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_file = tmp / "input.txt"
            input_file.write_text("hello from voxtral", encoding="utf-8")
            model_dir = tmp / "voxtral_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text('{"model_type":"voxtral_tts"}', encoding="utf-8")
            (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
            (model_dir / "tekken.json").write_text("{}", encoding="utf-8")

            model_store = ModelStore(root_dir=tmp / "models")
            options = RuntimeOptions(
                model_key="voxtral_tts",
                model_path_override=str(model_dir),
                voice="neutral_male",
                speed=None,
                lang_code=None,
                max_tokens=None,
                verbose=False,
                offline=True,
                audio_format="wav",
                parallel_workers=1,
            )

            runtime._LOADED_MODEL_CACHE.clear()
            with patch("libro_tts.runtime._get_load_model_fn", return_value=fake_load_model), patch(
                "libro_tts.runtime._get_audio_write_fn", return_value=fake_audio_write
            ), patch("libro_tts.runtime._print_user_message"), patch(
                "libro_tts.runtime.validate_model_load_preflight"
            ):
                result = process_single_file(
                    input_file=input_file,
                    output_prefix=str(tmp / "outs" / "voxtral"),
                    model_store=model_store,
                    options=options,
                    logger=__import__("logging").getLogger("test"),
                )

        self.assertEqual(result.files_processed, 1)
        self.assertEqual(captured_kwargs.get("voice"), "neutral_male")
        self.assertNotIn("ref_audio", captured_kwargs)
        self.assertNotIn("ref_text", captured_kwargs)


if __name__ == "__main__":
    unittest.main()
