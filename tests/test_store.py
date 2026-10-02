import tempfile
import unittest
from pathlib import Path
import json

from libro_tts.catalog import get_model_spec
from libro_tts.store import ModelStore


class StoreTests(unittest.TestCase):
    def test_first_run_download_then_local_reuse(self):
        spec = get_model_spec("kokoro")
        calls = {"count": 0}

        def fake_snapshot_download(repo_id: str, local_dir: str):
            calls["count"] += 1
            from pathlib import Path

            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors").write_text("weights", encoding="utf-8")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ModelStore(root_dir=Path(tmp_dir), snapshot_downloader=fake_snapshot_download)
            first = store.ensure_model(spec=spec, offline=False)
            second = store.ensure_model(spec=spec, offline=False)

        self.assertEqual(calls["count"], 1)
        self.assertEqual(first, second)

    def test_offline_fails_when_missing(self):
        spec = get_model_spec("dia")
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ModelStore(root_dir=Path(tmp_dir))
            with self.assertRaises(RuntimeError):
                store.ensure_model(spec=spec, offline=True)

    def test_stale_non_mlx_manifest_entry_is_ignored(self):
        spec = get_model_spec("kokoro")
        calls = {"count": 0}

        def fake_snapshot_download(repo_id: str, local_dir: str):
            calls["count"] += 1
            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors").write_text("weights", encoding="utf-8")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            store = ModelStore(root_dir=root, snapshot_downloader=fake_snapshot_download)
            stale_dir = root / "kokoro" / "prince-canuma__Kokoro-82M"
            stale_dir.mkdir(parents=True, exist_ok=True)
            (stale_dir / "config.json").write_text("{}", encoding="utf-8")
            (stale_dir / "model.safetensors").write_text("weights", encoding="utf-8")
            manifest = {
                "version": 1,
                "models": {
                    "kokoro": {
                        "repo_id": "prince-canuma/Kokoro-82M",
                        "path": str(stale_dir),
                    }
                },
            }
            (root / "resolved_models.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            result = store.ensure_model(spec=spec, offline=False)

        self.assertIn("mlx-community__Kokoro-82M-bf16", str(result))
        self.assertEqual(calls["count"], 1)

    def test_missing_model_type_is_backfilled_for_local_model(self):
        spec = get_model_spec("kokoro")

        def fake_snapshot_download(repo_id: str, local_dir: str):
            _ = repo_id
            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors").write_text("weights", encoding="utf-8")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            store = ModelStore(root_dir=root, snapshot_downloader=fake_snapshot_download)
            result = store.ensure_model(spec=spec, offline=False)

            config = json.loads((result / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(config.get("model_type"), "kokoro")

    def test_mismatched_model_type_is_normalized_to_spec_key(self):
        spec = get_model_spec("spark")

        def fake_snapshot_download(repo_id: str, local_dir: str):
            _ = repo_id
            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
            (target / "model.safetensors").write_text("weights", encoding="utf-8")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            store = ModelStore(root_dir=root, snapshot_downloader=fake_snapshot_download)
            result = store.ensure_model(spec=spec, offline=False)
            config = json.loads((result / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(config.get("model_type"), "spark")

    def test_csm_model_type_is_normalized_to_sesame(self):
        spec = get_model_spec("csm")

        def fake_snapshot_download(repo_id: str, local_dir: str):
            _ = repo_id
            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text('{"model_type":"csm"}', encoding="utf-8")
            (target / "model.safetensors").write_text("weights", encoding="utf-8")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            store = ModelStore(root_dir=root, snapshot_downloader=fake_snapshot_download)
            result = store.ensure_model(spec=spec, offline=False)
            config = json.loads((result / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(config.get("model_type"), "sesame")

    def test_voxtral_model_type_is_normalized_to_voxtral_tts(self):
        spec = get_model_spec("voxtral_tts")

        def fake_snapshot_download(repo_id: str, local_dir: str):
            _ = repo_id
            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
            (target / "model.safetensors").write_text("weights", encoding="utf-8")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            store = ModelStore(root_dir=root, snapshot_downloader=fake_snapshot_download)
            result = store.ensure_model(spec=spec, offline=False)
            config = json.loads((result / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(config.get("model_type"), "voxtral_tts")

    def test_incomplete_sharded_model_is_redownloaded(self):
        spec = get_model_spec("voxtral_tts")
        calls = {"count": 0}

        def fake_snapshot_download(repo_id: str, local_dir: str):
            calls["count"] += 1
            _ = repo_id
            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model-00001-of-00002.safetensors").write_text("weights-1", encoding="utf-8")
            (target / "model-00002-of-00002.safetensors").write_text("weights-2", encoding="utf-8")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stale_dir = root / "voxtral_tts" / "mlx-community__Voxtral-4B-TTS-2603-mlx-bf16"
            stale_dir.mkdir(parents=True, exist_ok=True)
            (stale_dir / "config.json").write_text("{}", encoding="utf-8")
            (stale_dir / "model-00002-of-00002.safetensors").write_text("weights-2", encoding="utf-8")
            (stale_dir / "voice_embedding").mkdir(parents=True, exist_ok=True)
            (stale_dir / "voice_embedding" / "neutral_female.safetensors").write_text(
                "embedding",
                encoding="utf-8",
            )

            store = ModelStore(root_dir=root, snapshot_downloader=fake_snapshot_download)
            result = store.ensure_model(spec=spec, offline=False)
            shard_one_exists = (result / "model-00001-of-00002.safetensors").exists()
            shard_two_exists = (result / "model-00002-of-00002.safetensors").exists()

        self.assertEqual(calls["count"], 1)
        self.assertTrue(shard_one_exists)
        self.assertTrue(shard_two_exists)

    def test_model_index_requires_all_referenced_weights(self):
        spec = get_model_spec("qwen3_tts")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "qwen3_tts" / "mlx-community__Qwen3-TTS-12Hz-1.7B-Base-bf16"
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                            "model.layers.0.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (target / "model-00001-of-00002.safetensors").write_text("weights-1", encoding="utf-8")

            store = ModelStore(root_dir=root)
            with self.assertRaises(RuntimeError):
                store.ensure_model(spec=spec, offline=True)

if __name__ == "__main__":
    unittest.main()
