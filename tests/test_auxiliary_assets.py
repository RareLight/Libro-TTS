import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from huggingface_hub import constants, hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
import numpy as np

from libro_tts.assets import (
    AssetSpec, CSM_CODEC, CSM_TOKENIZER, DIA_CODEC, CHATTERBOX_TOKENIZER, WHISPER,
    ensure_asset, local_asset_loading, prepare_model_assets,
)
import libro_tts.runtime as runtime
from libro_tts.store import COMPLETION_FILE, ModelStore

REVISION = "a" * 40


def cached_asset(root, spec, *, revision=REVISION):
    repo = root / ".hf/hub" / ("models--" + spec.repo_id.replace("/", "--"))
    target = repo / "snapshots" / revision
    for name in spec.required_assets:
        item = target / name
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("{}" if name.endswith(".json") else "weights")
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs/main").write_text(revision)
    return target


def spawned_cache_probe(root):
    root = Path(root)
    # The fresh worker starts with package bootstrap's project cache. The task
    # must select the caller's disposable root for both load and lazy lookup.
    def load(_reference):
        snapshot = Path(snapshot_download(DIA_CODEC.repo_id))
        assert snapshot.is_relative_to(root)
        return SimpleNamespace(generate=generate, sample_rate=24000)
    def generate(**_kwargs):
        asset = Path(hf_hub_download(DIA_CODEC.repo_id, "config.json"))
        assert asset.is_relative_to(root)
        yield SimpleNamespace(audio=np.array([0.1]), sample_rate=24000)
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network")), \
            patch("libro_tts.runtime._get_load_model_fn", return_value=load):
        index, audio, rate = runtime._parallel_worker_generate((0, "text", "fixture", {}, False, str(root / ".hf")))
    return index, len(audio), rate


class AuxiliaryAssetTests(unittest.TestCase):
    def test_existing_assets_reused_with_relative_inventory_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary), snapshot_downloader=Mock(side_effect=AssertionError("download")),
                               repo_info_loader=Mock(side_effect=AssertionError("metadata")))
            for spec in (CSM_TOKENIZER, CSM_CODEC, DIA_CODEC, CHATTERBOX_TOKENIZER, WHISPER):
                target = cached_asset(store.root_dir, spec)
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
                    self.assertEqual(ensure_asset(store, spec, True), target)
                entry = store._load_manifest()["assets"][spec.key]
                self.assertEqual(entry["revision"], REVISION)
                self.assertEqual(entry["scope"], "auxiliary_snapshot")
                self.assertFalse(Path(entry["path"]).is_absolute())
                self.assertEqual(entry["validation"], "local")

    def test_missing_and_incomplete_offline_assets_fail_before_acquisition(self):
        with tempfile.TemporaryDirectory() as temporary:
            downloader, metadata = Mock(), Mock()
            store = ModelStore(Path(temporary), snapshot_downloader=downloader, repo_info_loader=metadata)
            for spec in (CSM_TOKENIZER, CSM_CODEC, DIA_CODEC, CHATTERBOX_TOKENIZER, WHISPER):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
                    with self.assertRaisesRegex(RuntimeError, spec.key):
                        ensure_asset(store, spec, True)
                    target = cached_asset(store.root_dir, spec)
                    (target / spec.required_assets[0]).write_text("")
                    with self.assertRaisesRegex(RuntimeError, spec.key):
                        ensure_asset(store, spec, True)
            downloader.assert_not_called()
            metadata.assert_not_called()

    def test_selective_download_is_pinned_and_excludes_llama_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            metadata = Mock(return_value=SimpleNamespace(sha=REVISION, siblings=[
                SimpleNamespace(rfilename=name, size=size) for name, size in
                {"tokenizer.json": 2, "tokenizer_config.json": 2, "model.safetensors": 9_000_000_000}.items()
            ]))

            def download(**kwargs):
                calls.append(kwargs)
                root = Path(kwargs["local_dir"])
                for name in CSM_TOKENIZER.required_assets:
                    (root / name).write_text("{}")

            store = ModelStore(Path(temporary), snapshot_downloader=download, repo_info_loader=metadata)
            target = ensure_asset(store, CSM_TOKENIZER)
            self.assertEqual(calls[0]["revision"], REVISION)
            self.assertEqual(calls[0]["allow_patterns"], ["*.json"])
            self.assertNotIn("model.safetensors", json.loads((target / COMPLETION_FILE).read_text())["files"])
            metadata.reset_mock()
            self.assertEqual(ensure_asset(store, CSM_TOKENIZER, True), target)
            self.assertEqual(len(calls), 1)
            metadata.assert_not_called()

    def test_interrupted_acquisition_resumes_same_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Mock(return_value=SimpleNamespace(sha=REVISION, siblings=[
                SimpleNamespace(rfilename="config.json", size=2),
                SimpleNamespace(rfilename="model.safetensors", size=7),
            ]))
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                root = Path(kwargs["local_dir"])
                (root / "model.safetensors").write_text("weights")
                if len(calls) == 1:
                    raise RuntimeError("interrupted")
                self.assertTrue((root / "model.safetensors").exists())
                (root / "config.json").write_text("{}")

            store = ModelStore(Path(temporary), snapshot_downloader=download, repo_info_loader=metadata)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                ensure_asset(store, DIA_CODEC)
            self.assertFalse(store.manifest_path.exists())
            result = ensure_asset(store, DIA_CODEC)
            self.assertTrue(result.exists())
            metadata.assert_called_once()
            self.assertEqual([item["revision"] for item in calls], [REVISION, REVISION])

    def test_truncated_completed_asset_is_rejected_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary))
            target = cached_asset(store.root_dir, DIA_CODEC)
            ensure_asset(store, DIA_CODEC, True)
            (target / "model.safetensors").write_text("short")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                ensure_asset(store, DIA_CODEC, True)

    def test_cache_blob_symlinks_are_portable_and_external_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "original"
            store = ModelStore(original)
            target = cached_asset(original, CSM_CODEC)
            tensor = target / CSM_CODEC.required_assets[0]
            blob = target.parent.parent / "blobs/tensor"
            blob.parent.mkdir()
            tensor.replace(blob)
            tensor.symlink_to(os.path.relpath(blob, tensor.parent))
            ensure_asset(store, CSM_CODEC, True)
            relocated = Path(temporary) / "relocated"
            shutil.copytree(original, relocated, symlinks=True)
            shutil.rmtree(original)
            destination_store = ModelStore(relocated)
            self.assertTrue(ensure_asset(destination_store, CSM_CODEC, True).is_relative_to(relocated.resolve()))
            outside = Path(temporary) / "outside"
            outside.write_text("weights")
            moved_tensor = relocated / tensor.relative_to(original)
            moved_tensor.unlink()
            moved_tensor.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                ensure_asset(destination_store, CSM_CODEC, True)

    def test_real_hf_lookup_uses_managed_cache_and_restores_policy_on_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary))
            target = cached_asset(store.root_dir, DIA_CODEC)
            ensure_asset(store, DIA_CODEC, True)
            before = (constants.HF_HUB_CACHE, constants.HF_HUB_OFFLINE, os.environ.get("HF_HUB_OFFLINE"))
            with patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(ValueError, "scope"):
                    with local_asset_loading(store.root_dir / ".hf"):
                        self.assertEqual(Path(snapshot_download(DIA_CODEC.repo_id)), target)
                        self.assertEqual(Path(hf_hub_download(DIA_CODEC.repo_id, "config.json")), target / "config.json")
                        with self.assertRaises(LocalEntryNotFoundError):
                            hf_hub_download(DIA_CODEC.repo_id, "missing.json")
                        raise ValueError("scope")
            self.assertEqual(before, (constants.HF_HUB_CACHE, constants.HF_HUB_OFFLINE, os.environ.get("HF_HUB_OFFLINE")))

    def test_inherited_offline_policy_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(constants, "HF_HUB_OFFLINE", True):
            store = ModelStore(Path(temporary), repo_info_loader=Mock())
            with self.assertRaisesRegex(RuntimeError, "missing"):
                ensure_asset(store, DIA_CODEC, False)
            store._repo_info_loader.assert_not_called()
            with local_asset_loading(store.root_dir / ".hf"):
                self.assertTrue(constants.HF_HUB_OFFLINE)
            self.assertTrue(constants.HF_HUB_OFFLINE)

    def test_custom_csm_tokenizer_repository_is_managed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary))
            model = store.root_dir / "model"
            model.mkdir()
            (model / "config.json").write_text('{"text_tokenizer":"example/tokenizer"}')
            custom = AssetSpec("csm_text", "example/tokenizer", CSM_TOKENIZER.required_assets, ("*.json",))
            target = cached_asset(store.root_dir, custom)
            cached_asset(store.root_dir, CSM_CODEC)
            prepare_model_assets(store, "csm", str(model), True)
            self.assertEqual(store._load_manifest()["assets"]["csm_text"]["path"], target.relative_to(store.root_dir).as_posix())

    def test_whisper_loads_local_path_with_supported_api_and_caches_by_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime._LOADED_STT_MODEL = None
            runtime._LOADED_STT_REFERENCE = None
            loader = Mock(return_value=SimpleNamespace(generate=Mock(return_value=SimpleNamespace(text=" transcript "))))
            try:
                with patch("libro_tts.runtime._get_stt_load_fn", return_value=loader):
                    for root_name in ("first", "second"):
                        store = ModelStore(Path(temporary) / root_name)
                        target = cached_asset(store.root_dir, WHISPER)
                        for _ in range(2):
                            value = runtime._auto_transcribe_reference_text(ref_audio_path="reference.wav",
                                logger=logging.getLogger("test"), verbose=False, model_store=store, offline=True)
                            self.assertEqual(value, "transcript")
                        self.assertEqual(loader.call_args.args, (str(target),))
                    self.assertEqual(loader.call_count, 2)
            finally:
                runtime._LOADED_STT_MODEL = None
                runtime._LOADED_STT_REFERENCE = None

    def test_spawn_worker_uses_passed_cache_for_load_and_lazy_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / ".hf"
            expected = str(cache.resolve() / "hub")
            def load(_reference):
                self.assertEqual(constants.HF_HUB_CACHE, expected)
                self.assertTrue(constants.HF_HUB_OFFLINE)
                return SimpleNamespace(generate=generate, sample_rate=24000)
            def generate(**_kwargs):
                self.assertEqual(constants.HF_HUB_CACHE, expected)
                self.assertTrue(constants.HF_HUB_OFFLINE)
                yield SimpleNamespace(audio=np.array([0.1]), sample_rate=24000)
            runtime._PARALLEL_WORKER_MODEL = None
            runtime._LOADED_MODEL_CACHE.clear()
            try:
                with patch("libro_tts.runtime._get_load_model_fn", return_value=load):
                    index, audio, rate = runtime._parallel_worker_generate((3, "text", "fixture", {}, False, str(cache)))
                    self.assertEqual((index, rate, len(audio)), (3, 24000, 1))
            finally:
                runtime._PARALLEL_WORKER_MODEL = None
                runtime._LOADED_MODEL_CACHE.clear()

    def test_fresh_spawn_process_resolves_only_callers_managed_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            store = ModelStore(root)
            cached_asset(root, DIA_CODEC)
            ensure_asset(store, DIA_CODEC, True)
            with mp.get_context("spawn").Pool(1) as pool:
                self.assertEqual(pool.apply_async(spawned_cache_probe, (str(root),)).get(timeout=15), (0, 1, 24000))

    def test_failed_repair_preserves_previous_snapshot_and_pinned_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = cached_asset(root, DIA_CODEC)
            store = ModelStore(root)
            ensure_asset(store, DIA_CODEC, True)
            (target / "config.json").write_text("")
            metadata = Mock(return_value=SimpleNamespace(sha=REVISION, siblings=[
                SimpleNamespace(rfilename="config.json", size=2),
                SimpleNamespace(rfilename="model.safetensors", size=7),
            ]))
            def interrupt(**kwargs):
                self.assertEqual(kwargs["revision"], REVISION)
                raise RuntimeError("interrupted repair")
            store = ModelStore(root, snapshot_downloader=interrupt, repo_info_loader=metadata)
            with self.assertRaisesRegex(RuntimeError, "interrupted repair"):
                ensure_asset(store, DIA_CODEC)
            metadata.assert_called_once_with(repo_id=DIA_CODEC.repo_id, files_metadata=True, revision=REVISION)
            self.assertEqual((target / "model.safetensors").read_text(), "weights")
            self.assertEqual((target.parent.parent / "refs/main").read_text(), REVISION)

    def test_completion_revision_mismatch_and_malformed_asset_manifest_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary))
            target = cached_asset(store.root_dir, DIA_CODEC)
            ensure_asset(store, DIA_CODEC, True)
            marker = target / COMPLETION_FILE
            payload = json.loads(marker.read_text())
            payload["revision"] = "b" * 40
            marker.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                ensure_asset(store, DIA_CODEC, True)
            manifest = store._load_manifest()
            manifest["assets"][DIA_CODEC.key] = {"revision": REVISION}
            store.manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                ensure_asset(store, DIA_CODEC, True)

    def test_acquisition_selects_all_cache_roots_and_restores_them_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ModelStore(Path(temporary))
            before = constants.HF_XET_CACHE, os.environ.get("HF_TOKEN_PATH"), constants.HF_HUB_CACHE
            def metadata(**_kwargs):
                self.assertEqual(Path(constants.HF_XET_CACHE), store.root_dir / ".hf/xet")
                self.assertEqual(Path(constants.HF_TOKEN_PATH), store.root_dir / ".hf/token")
                self.assertFalse(constants.HF_HUB_OFFLINE)
                raise RuntimeError("metadata failed")
            store._repo_info_loader = metadata
            with self.assertRaisesRegex(RuntimeError, "metadata failed"):
                ensure_asset(store, DIA_CODEC)
            self.assertEqual(before, (constants.HF_XET_CACHE, os.environ.get("HF_TOKEN_PATH"), constants.HF_HUB_CACHE))

    def test_tts_runtime_cache_does_not_reuse_auxiliaries_from_another_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            loader = Mock(side_effect=[object(), object()])
            runtime._LOADED_MODEL_CACHE.clear()
            try:
                with patch("libro_tts.runtime._get_load_model_fn", return_value=loader):
                    results = []
                    for name in ("first", "second"):
                        with local_asset_loading(Path(temporary) / name / ".hf"):
                            model = runtime._load_runtime_model("same_primary_path", logging.getLogger("test"))
                            self.assertIs(runtime._load_runtime_model("same_primary_path", logging.getLogger("test")), model)
                            results.append(model)
                    self.assertIsNot(results[0], results[1])
                    self.assertEqual(loader.call_count, 2)
            finally:
                runtime._LOADED_MODEL_CACHE.clear()


if __name__ == "__main__":
    unittest.main()
