from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libro_tts.runtime import _resolve_local_kokoro_voice, _validate_voxtral_voice
from libro_tts.store import ModelStore
from libro_tts.assets import AssetSpec
from tests.test_auxiliary_assets import cached_asset


class KokoroAssetsTests(unittest.TestCase):
    def test_named_voice_and_blend_use_included_local_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            voices = root / "voices"
            voices.mkdir()
            first = voices / "af_aoede.safetensors"
            second = voices / "af_bella.safetensors"
            first.write_text("voice")
            second.write_text("voice")
            self.assertEqual(_resolve_local_kokoro_voice(str(root), "af_aoede", True), str(first.resolve()))
            self.assertEqual(
                _resolve_local_kokoro_voice(str(root), "af_aoede,af_bella", True),
                f"{first.resolve()},{second.resolve()}",
            )

    def test_missing_offline_voice_fails_with_local_asset_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "missing locally"):
                _resolve_local_kokoro_voice(temporary, "af_aoede", True)

    def test_missing_primary_voice_reuses_managed_auxiliary_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary) / "models")
            spec = AssetSpec("kokoro_voice_af_aoede", "prince-canuma/Kokoro-82M",
                             ("voices/af_aoede.safetensors",), ("voices/af_aoede.safetensors",))
            target = cached_asset(store.root_dir, spec)
            self.assertEqual(_resolve_local_kokoro_voice(temporary, "af_aoede", True, store),
                             str(target / spec.required_assets[0]))

    def test_online_missing_voice_acquires_managed_asset_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary) / "models")
            tensor = Path(temporary) / "acquired/voices/af_aoede.safetensors"
            with patch("libro_tts.runtime.ensure_asset", return_value=tensor.parent.parent) as acquire:
                self.assertEqual(_resolve_local_kokoro_voice(temporary, "af_aoede", False, store), str(tensor.resolve()))
                self.assertEqual(acquire.call_args.args[1].allow_patterns, ("voices/af_aoede.safetensors",))

    def test_explicit_voice_file_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            voice = Path(temporary) / "custom.safetensors"
            voice.write_text("voice")
            self.assertEqual(_resolve_local_kokoro_voice(temporary, str(voice), True), str(voice.resolve()))

    def test_empty_explicit_tensor_and_invalid_named_voice_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            voice = Path(temporary) / "empty.safetensors"
            voice.touch()
            with self.assertRaisesRegex(RuntimeError, "missing locally"):
                _resolve_local_kokoro_voice(temporary, str(voice), False)
            with self.assertRaisesRegex(RuntimeError, "Invalid"):
                _resolve_local_kokoro_voice(temporary, "../escape", False, ModelStore(Path(temporary) / "models"))

    def test_voxtral_selected_voice_requires_nonempty_contained_embedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "embedding"):
                _validate_voxtral_voice(temporary, "cheerful_female")
            (root / "voice_embedding").mkdir()
            embedding = root / "voice_embedding/cheerful_female.safetensors"
            embedding.write_text("voice")
            _validate_voxtral_voice(temporary, "cheerful_female")
            embedding.write_text("")
            with self.assertRaisesRegex(RuntimeError, "embedding"):
                _validate_voxtral_voice(temporary, "cheerful_female")


if __name__ == "__main__":
    unittest.main()
