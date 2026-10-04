import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libro_tts.catalog import get_model_spec
from libro_tts.runtime import (
    DEFAULT_PARALLEL_MAX_CHUNKS,
    DEFAULT_PARALLEL_WORKERS,
    SPARK_MAX_REF_TEXT_CHARS,
    MIN_PARALLEL_CHUNK_COUNT,
    RuntimeOptions,
    _chunk_text_for_generation,
    _model_supports_parallel_chunk_generation,
    _parallel_chunk_batch_size,
    _populate_missing_ref_text,
    _should_parallelize_chunk_generation,
    _validate_local_model_path,
    _resolve_output_path,
    _resolve_model_reference,
    default_output_target_for_input,
    build_generate_kwargs,
)
from libro_tts.store import ModelStore


class RuntimeDefaultTests(unittest.TestCase):
    def setUp(self):
        self.reference_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        voices = self.reference_root / "reference_voices"
        voices.mkdir()
        for name in ("Britney", "Emma"):
            (voices / f"{name}.wav").write_bytes(b"synthetic reference; synthesis is mocked")
            (voices / f"{name}.txt").write_text("Synthetic reference transcript for tests.")
        self.enterContext(patch("libro_tts.runtime.project_root", return_value=self.reference_root))

    def test_validate_local_model_path_accepts_csm_alias_model_type(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "csm_local"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text('{"model_type":"sesame"}', encoding="utf-8")
            (model_dir / "weights.safetensors").write_text("weights", encoding="utf-8")

            _validate_local_model_path(model_dir, "csm", logging.getLogger("test"))

    def test_kokoro_defaults_when_unspecified(self):
        spec = get_model_spec("kokoro")
        kwargs = build_generate_kwargs(
            text="Hello",
            spec=spec,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
        )
        self.assertEqual(kwargs["voice"], "af_aoede")
        self.assertAlmostEqual(kwargs["speed"], 1.1)
        self.assertEqual(kwargs["lang_code"], "a")

    def test_spark_speed_normalization(self):
        spec = get_model_spec("spark")
        kwargs = build_generate_kwargs(
            text="Hello",
            spec=spec,
            voice=None,
            speed=1.1,
            lang_code=None,
            max_tokens=None,
            verbose=False,
        )
        self.assertEqual(kwargs["speed"], 1.0)

    def test_qwen_defaults_use_vivian_and_english(self):
        spec = get_model_spec("qwen3_tts")
        kwargs = build_generate_kwargs(
            text="Hello",
            spec=spec,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
        )
        self.assertEqual(kwargs["voice"], "Vivian")
        self.assertEqual(kwargs["lang_code"], "english")

    def test_voxtral_defaults_use_neutral_female_without_speed_or_lang(self):
        spec = get_model_spec("voxtral_tts")
        kwargs = build_generate_kwargs(
            text="Hello",
            spec=spec,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
        )
        self.assertEqual(kwargs["voice"], "neutral_female")
        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertNotIn("speed", kwargs)
        self.assertNotIn("lang_code", kwargs)

    def test_prompt_models_default_to_britney_voice_prompt(self):
        spec = get_model_spec("csm")
        kwargs = build_generate_kwargs(
            text="Hello",
            spec=spec,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
        )
        expected_audio = self.reference_root / "reference_voices" / "Britney.wav"
        expected_text = (self.reference_root / "reference_voices" / "Britney.txt").read_text(
            encoding="utf-8"
        ).strip()
        self.assertNotIn("ref_text", kwargs)
        _populate_missing_ref_text(model_key="csm", kwargs=kwargs, logger=logging.getLogger("test"), verbose=False)
        self.assertEqual(kwargs["ref_audio"], str(expected_audio.resolve()))
        self.assertEqual(kwargs["ref_text"], " ".join(expected_text.split()))
        self.assertNotIn("voice", kwargs)
        self.assertEqual(kwargs["lang_code"], "en")

    def test_prompt_model_voice_name_override_resolves_in_reference_voices(self):
        spec = get_model_spec("dia")
        kwargs = build_generate_kwargs(
            text="Hello",
            spec=spec,
            voice="Britney",
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
        )
        expected_audio = self.reference_root / "reference_voices" / "Britney.wav"
        expected_text = (self.reference_root / "reference_voices" / "Britney.txt").read_text(
            encoding="utf-8"
        ).strip()
        self.assertNotIn("ref_text", kwargs)
        _populate_missing_ref_text(model_key="dia", kwargs=kwargs, logger=logging.getLogger("test"), verbose=False)
        self.assertEqual(kwargs["ref_audio"], str(expected_audio.resolve()))
        self.assertEqual(kwargs["ref_text"], " ".join(expected_text.split()))
        self.assertNotIn("voice", kwargs)

    def test_prompt_model_missing_voice_prompt_raises(self):
        spec = get_model_spec("spark")
        with self.assertRaises(FileNotFoundError):
            build_generate_kwargs(
                text="Hello",
                spec=spec,
                voice="missing-voice-prompt-does-not-exist",
                speed=None,
                lang_code=None,
                max_tokens=None,
                verbose=False,
            )

    def test_populate_missing_ref_text_uses_existing_transcript(self):
        kwargs = {
            "ref_audio": str((self.reference_root / "reference_voices" / "Britney.wav").resolve()),
        }
        _populate_missing_ref_text(
            model_key="dia",
            kwargs=kwargs,
            logger=logging.getLogger("test"),
            verbose=False,
        )
        self.assertIn("ref_text", kwargs)
        self.assertTrue(kwargs["ref_text"].strip())

    @patch("libro_tts.runtime._auto_transcribe_reference_text", return_value="auto transcript")
    def test_populate_missing_ref_text_falls_back_to_auto_transcription(self, mock_transcribe):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_audio = Path(tmp_dir) / "speaker.wav"
            ref_audio.write_bytes(b"fake")

            kwargs = {"ref_audio": str(ref_audio)}
            _populate_missing_ref_text(
                model_key="spark",
                kwargs=kwargs,
                logger=logging.getLogger("test"),
                verbose=False,
            )

        self.assertEqual(kwargs["ref_text"], "auto transcript")
        mock_transcribe.assert_called_once()

    @patch(
        "libro_tts.runtime._auto_transcribe_reference_text",
        return_value="synthetic spark transcript for testing.",
    )
    def test_spark_long_file_transcript_prefers_stt_and_caps_length(self, mock_transcribe):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_audio = Path(tmp_dir) / "speaker.wav"
            ref_audio.write_bytes(b"fake")
            (Path(tmp_dir) / "speaker.txt").write_text("word " * 500, encoding="utf-8")

            kwargs = {"ref_audio": str(ref_audio)}
            _populate_missing_ref_text(
                model_key="spark",
                kwargs=kwargs,
                logger=logging.getLogger("test"),
                verbose=False,
            )

        self.assertEqual(kwargs["ref_text"], "synthetic spark transcript for testing.")
        self.assertLessEqual(len(kwargs["ref_text"]), SPARK_MAX_REF_TEXT_CHARS)
        mock_transcribe.assert_called_once()

    def test_spark_existing_ref_text_is_trimmed(self):
        kwargs = {"ref_audio": "dummy.wav", "ref_text": "x" * (SPARK_MAX_REF_TEXT_CHARS + 120)}
        _populate_missing_ref_text(
            model_key="spark",
            kwargs=kwargs,
            logger=logging.getLogger("test"),
            verbose=False,
        )
        self.assertLessEqual(len(kwargs["ref_text"]), SPARK_MAX_REF_TEXT_CHARS)

    @patch("libro_tts.runtime._auto_transcribe_reference_text", return_value="aligned transcript")
    def test_spark_normal_builder_preserves_long_file_recovery(self, transcribe):
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "speaker.wav"
            audio.write_bytes(b"audio")
            audio.with_suffix(".txt").write_text("long transcript " * 100)
            kwargs = build_generate_kwargs(text="Hello", spec=get_model_spec("spark"), voice=str(audio),
                                           speed=None, lang_code=None, max_tokens=None, verbose=False)
            _populate_missing_ref_text(model_key="spark", kwargs=kwargs, logger=logging.getLogger("test"), verbose=False)
            self.assertEqual(kwargs["ref_text"], "aligned transcript")
            transcribe.assert_called_once()

    def test_model_path_override_rejects_non_mlx_repo_id(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override="prince-canuma/Kokoro-82M",
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ModelStore(root_dir=Path(tmp_dir))
            with self.assertRaises(RuntimeError):
                _resolve_model_reference(store, options, logging.getLogger("test"))

    def test_model_path_override_accepts_mlx_repo_id(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override="mlx-community/Kokoro-82M-bf16",
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ModelStore(root_dir=Path(tmp_dir))
            with patch.object(store, "ensure_model", return_value=Path(tmp_dir) / "managed") as ensure:
                model_reference, _spec = _resolve_model_reference(store, options, logging.getLogger("test"))
                self.assertEqual(ensure.call_args.kwargs["spec"].repo_candidates, (options.model_path_override,))
                self.assertFalse(ensure.call_args.kwargs["offline"])

        self.assertEqual(model_reference, str(Path(tmp_dir) / "managed"))
        self.assertIsNone(_spec)

    def test_offline_repo_override_uses_managed_store_and_propagates_missing_assets(self):
        options = RuntimeOptions("kokoro", "mlx-community/Kokoro-82M-bf16", None, None, None, None, False, True)
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary))
            with patch.object(store, "ensure_model", side_effect=RuntimeError("no complete local primary snapshot")) as ensure:
                with self.assertRaisesRegex(RuntimeError, "no complete local"):
                    _resolve_model_reference(store, options, logging.getLogger("test"))
                self.assertTrue(ensure.call_args.kwargs["offline"])
                self.assertEqual(ensure.call_args.kwargs["spec"].repo_candidates, (options.model_path_override,))

    def test_local_model_path_override_requires_config(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override=None,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "local_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "weights.safetensors").write_text("weights", encoding="utf-8")
            options.model_path_override = str(model_dir)
            store = ModelStore(root_dir=Path(tmp_dir) / "models")
            with self.assertRaises(RuntimeError):
                _resolve_model_reference(store, options, logging.getLogger("test"))

    def test_local_model_path_override_requires_safetensors(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override=None,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "local_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            options.model_path_override = str(model_dir)
            store = ModelStore(root_dir=Path(tmp_dir) / "models")
            with self.assertRaises(RuntimeError):
                _resolve_model_reference(store, options, logging.getLogger("test"))

    def test_model_path_override_uses_only_explicit_voice_speed_lang(self):
        kwargs = build_generate_kwargs(
            text="Hello",
            spec=None,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
        )
        self.assertNotIn("voice", kwargs)
        self.assertNotIn("speed", kwargs)
        self.assertNotIn("lang_code", kwargs)

    def test_output_path_respects_existing_extension(self):
        path_with_ext = _resolve_output_path("chapter.wav", "wav")
        path_without_ext = _resolve_output_path("chapter", "wav")
        path_mismatch = _resolve_output_path("chapter.mp3", "wav")

        self.assertEqual(str(path_with_ext), "chapter.wav")
        self.assertEqual(str(path_without_ext), "chapter.wav")
        self.assertEqual(str(path_mismatch), "chapter.wav")

    def test_default_output_target_avoids_double_audio_extension(self):
        target = default_output_target_for_input(
            input_file=Path("foo.wav.txt"),
            audio_format="wav",
        )
        self.assertEqual(target, "foo.wav")

    def test_default_output_target_sanitizes_hidden_txt_name(self):
        target = default_output_target_for_input(
            input_file=Path(".txt"),
            audio_format="wav",
        )
        self.assertEqual(target, "txt.wav")

    def test_chunker_targets_reasonable_chunk_sizes(self):
        text = " ".join(
            [
                "This is a sentence for chunking."
                " We need robust segmentation for reliable generation."
                " The runtime should prefer chunks around ten to fifteen seconds."
            ]
            * 20
        )
        chunks = _chunk_text_for_generation(text, speed=1.0)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks[:-1]:
            self.assertLessEqual(len(chunk), 180)
            self.assertGreaterEqual(len(chunk), 80)

    def test_chunker_preserves_short_text_without_splitting(self):
        text = "Short test sentence."
        chunks = _chunk_text_for_generation(text, speed=1.0)
        self.assertEqual(chunks, [text])

    def test_runtime_parallel_defaults(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override=None,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
        )
        self.assertEqual(options.parallel_workers, DEFAULT_PARALLEL_WORKERS)
        self.assertEqual(options.parallel_max_chunks, DEFAULT_PARALLEL_MAX_CHUNKS)

    def test_parallel_chunking_disabled_for_small_chunk_counts(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override=None,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
        )
        self.assertFalse(_should_parallelize_chunk_generation(options, MIN_PARALLEL_CHUNK_COUNT - 1))
        self.assertTrue(_should_parallelize_chunk_generation(options, MIN_PARALLEL_CHUNK_COUNT))

    def test_parallel_chunking_disabled_for_models_with_reliability_guard(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override=None,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
        )
        self.assertTrue(_model_supports_parallel_chunk_generation("kokoro"))
        self.assertFalse(_model_supports_parallel_chunk_generation("csm"))
        self.assertFalse(_model_supports_parallel_chunk_generation("dia"))
        self.assertFalse(
            _should_parallelize_chunk_generation(
                options,
                chunk_count=12,
                model_key="csm",
            )
        )
        self.assertFalse(
            _should_parallelize_chunk_generation(
                options,
                chunk_count=12,
                model_key="dia",
            )
        )

    def test_parallel_chunk_batch_size_uses_default_cap(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override=None,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
        )
        self.assertEqual(_parallel_chunk_batch_size(options, 5), 5)
        self.assertEqual(_parallel_chunk_batch_size(options, 50), DEFAULT_PARALLEL_MAX_CHUNKS)

    def test_parallel_chunk_batch_size_uses_all_chunks_when_nonpositive(self):
        options = RuntimeOptions(
            model_key="kokoro",
            model_path_override=None,
            voice=None,
            speed=None,
            lang_code=None,
            max_tokens=None,
            verbose=False,
            offline=False,
            audio_format="wav",
            parallel_max_chunks=0,
        )
        self.assertEqual(_parallel_chunk_batch_size(options, 7), 7)


if __name__ == "__main__":
    unittest.main()
