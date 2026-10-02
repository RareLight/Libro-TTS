import unittest

from libro_tts.catalog import (
    CATALOG,
    DEFAULT_MODEL_KEY,
    KOKORO_ENGLISH_VOICES,
    REQUIRED_MODEL_KEYS,
    VOXTRAL_ENGLISH_VOICES,
    accepted_model_types_for_key,
    canonical_model_type_for_key,
    get_model_spec,
    is_mlx_repo_id,
    list_kokoro_english_voices,
    list_voxtral_english_voices,
    resolve_model_key,
)


class CatalogTests(unittest.TestCase):
    def test_catalog_has_exact_required_keys(self):
        self.assertEqual(set(CATALOG.keys()), set(REQUIRED_MODEL_KEYS))

    def test_default_model_exists(self):
        self.assertIn(DEFAULT_MODEL_KEY, CATALOG)

    def test_each_model_has_repo_candidates(self):
        for key in REQUIRED_MODEL_KEYS:
            spec = get_model_spec(key)
            self.assertGreaterEqual(len(spec.repo_candidates), 1)

    def test_repo_candidates_are_mlx_variants_only(self):
        for key in REQUIRED_MODEL_KEYS:
            spec = get_model_spec(key)
            self.assertTrue(all(is_mlx_repo_id(repo) for repo in spec.repo_candidates))

    def test_spark_repo_points_to_current_bf16_variant(self):
        spark = get_model_spec("spark")
        self.assertEqual(spark.repo_candidates[0], "mlx-community/Spark-TTS-0.5B-bf16")

    def test_csm_canonical_model_type_uses_sesame(self):
        self.assertEqual(canonical_model_type_for_key("csm"), "sesame")
        self.assertEqual(accepted_model_types_for_key("csm"), ("sesame", "csm"))

    def test_model_aliases_resolve_to_canonical_keys(self):
        self.assertEqual(resolve_model_key("qwen"), "qwen3_tts")
        self.assertEqual(resolve_model_key("qwen3-tts"), "qwen3_tts")
        self.assertEqual(resolve_model_key("sesame"), "csm")
        self.assertEqual(resolve_model_key("spark_tts"), "spark")
        self.assertEqual(resolve_model_key("voxtral"), "voxtral_tts")
        self.assertEqual(resolve_model_key("voxtral-tts"), "voxtral_tts")

    def test_every_required_key_resolves_to_itself(self):
        for key in REQUIRED_MODEL_KEYS:
            self.assertEqual(resolve_model_key(key), key)

    def test_kokoro_english_voice_list(self):
        voices = list_kokoro_english_voices()
        self.assertEqual(voices, KOKORO_ENGLISH_VOICES)
        self.assertEqual(len(voices), 28)
        self.assertIn("af_aoede", voices)
        self.assertIn("bm_lewis", voices)

    def test_voxtral_repo_points_to_current_bf16_variant(self):
        voxtral = get_model_spec("voxtral_tts")
        self.assertEqual(
            voxtral.repo_candidates[0],
            "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
        )
        self.assertEqual(voxtral.default_voice, "neutral_female")

    def test_voxtral_english_voice_list(self):
        voices = list_voxtral_english_voices()
        self.assertEqual(voices, VOXTRAL_ENGLISH_VOICES)
        self.assertEqual(len(voices), 5)
        self.assertIn("casual_male", voices)
        self.assertIn("neutral_female", voices)


if __name__ == "__main__":
    unittest.main()
