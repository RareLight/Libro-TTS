from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from libro_tts.catalog import get_model_spec, list_model_specs
from libro_tts.store import COMPLETION_FILE, ModelStore
from tests.test_store import write_assets


REVISION = "a" * 40


def repo_info(files=None, revision=REVISION):
    return SimpleNamespace(
        sha=revision,
        siblings=[SimpleNamespace(rfilename=name, size=size)
                  for name, size in (files or {"config.json": None}).items()],
    )


def write_snapshot(target, spec, *, assets=True):
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_text("weights")
    if assets:
        write_assets(spec, target)


def concurrent_writer(root, key, start, results):
    try:
        store = ModelStore(root_dir=Path(root))
        original_load = store._load_manifest

        def slow_load():
            data = original_load()
            # Enlarge the lost-update window. The shared transaction lock must
            # cover this read and its following replacement across processes.
            time.sleep(0.08)
            return data

        store._load_manifest = slow_load
        if not start.wait(10):
            raise RuntimeError("Writer start timed out")
        store.ensure_model(get_model_spec(key), offline=True)
        results.put((key, None))
    except Exception as exc:
        results.put((key, repr(exc)))


class StoreRecoveryTests(unittest.TestCase):
    def make_store(self, root, downloader=None, **kwargs):
        kwargs.setdefault("repo_info_loader", Mock(return_value=repo_info()))
        return ModelStore(root_dir=root, snapshot_downloader=downloader, **kwargs)

    def test_all_catalog_primary_assets_are_required_for_local_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            for spec in list_model_specs():
                with self.subTest(model=spec.key):
                    target = Path(temporary) / spec.key
                    write_snapshot(target, spec, assets=False)
                    self.assertEqual(ModelStore._has_ready_model(target, spec), not spec.required_assets)
                    write_assets(spec, target)
                    self.assertTrue(ModelStore._has_ready_model(target, spec))
                    for pattern in spec.required_assets:
                        asset = next(target.glob(pattern))
                        content = asset.read_bytes()
                        asset.unlink()
                        self.assertFalse(ModelStore._has_ready_model(target, spec), pattern)
                        asset.write_bytes(content)

    def test_interruption_resumes_same_revision_without_exposing_partial_snapshot(self):
        spec = get_model_spec("voxtral_tts")
        calls = []
        loader = Mock(return_value=repo_info())

        def download(repo_id, local_dir, revision):
            target = Path(local_dir)
            calls.append((repo_id, target, revision))
            if len(calls) == 1:
                write_snapshot(target, spec, assets=False)
                partial = target / ".cache/huggingface/download/unfinished.incomplete"
                partial.parent.mkdir(parents=True)
                partial.write_text("resumable progress")
                raise ConnectionError("interrupted before tokenizer")
            self.assertEqual((target / "model.safetensors").read_text(), "weights")
            self.assertEqual((target / ".cache/huggingface/download/unfinished.incomplete").read_text(), "resumable progress")
            write_assets(spec, target)
            return str(target)

        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary), download, repo_info_loader=loader)
            published = store._target_dir(spec.key, spec.repo_candidates[0])
            with self.assertRaisesRegex(RuntimeError, "interrupted before tokenizer"):
                store.ensure_model(spec)
            self.assertFalse(published.exists())
            with self.assertRaisesRegex(RuntimeError, "offline mode"):
                store.ensure_model(spec, offline=True)
            self.assertEqual(len(calls), 1)
            loader.return_value = repo_info(revision="b" * 40)
            result = store.ensure_model(spec)
            self.assertEqual(result, published)
            self.assertEqual(calls[0], calls[1])
            loader.assert_called_once()
            marker = json.loads((result / COMPLETION_FILE).read_text())
            self.assertEqual(marker["revision"], REVISION)
            self.assertEqual(marker["scope"], "primary_snapshot")
            self.assertEqual(marker["validation"], "downloaded")
            entry = store._load_manifest()["models"][spec.key]
            self.assertTrue(entry["complete"])
            self.assertFalse(Path(entry["path"]).is_absolute())
            self.assertEqual(store.ensure_model(spec, offline=True), result)

    def test_incomplete_legacy_snapshot_preserves_weights_and_hf_progress_on_repair(self):
        spec = get_model_spec("kokoro")

        def download(repo_id, local_dir, revision):
            target = Path(local_dir)
            self.assertEqual((target / "model.safetensors").read_text(), "weights")
            self.assertEqual((target / ".cache/partial.incomplete").read_text(), "progress")
            write_assets(spec, target)

        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary), download)
            target = store._target_dir(spec.key, spec.repo_candidates[0])
            write_snapshot(target, spec, assets=False)
            (target / ".cache").mkdir()
            (target / ".cache/partial.incomplete").write_text("progress")
            with self.assertRaisesRegex(RuntimeError, "offline mode"):
                store.ensure_model(spec, offline=True)
            self.assertEqual(store.ensure_model(spec), target)
            self.assertTrue((target / ".cache/partial.incomplete").is_file())

    def test_returned_but_truncated_download_is_not_published(self):
        spec = get_model_spec("dia")
        loader = Mock(return_value=repo_info({"config.json": 2, "model.safetensors": 8}))

        def download(repo_id, local_dir, revision):
            write_snapshot(Path(local_dir), spec)

        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary), download, repo_info_loader=loader)
            with self.assertRaisesRegex(RuntimeError, "truncated 'model.safetensors'"):
                store.ensure_model(spec)
            self.assertFalse(store._target_dir(spec.key, spec.repo_candidates[0]).exists())
            self.assertFalse(store.manifest_path.exists())

    def test_missing_asset_is_repaired_even_when_download_returns_normally(self):
        spec = get_model_spec("soprano")
        calls = []

        def download(repo_id, local_dir, revision):
            calls.append(local_dir)
            write_snapshot(Path(local_dir), spec, assets=len(calls) > 1)

        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary), download)
            with self.assertRaisesRegex(RuntimeError, "required assets"):
                store.ensure_model(spec)
            result = store.ensure_model(spec)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], calls[1])
            self.assertTrue((result / "tokenizer.json").is_file())

    def test_absent_or_malformed_weights_and_config_are_not_ready(self):
        spec = get_model_spec("dia")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            for config, weights in (("{", "weights"), ("[]", "weights"), ("{}", "")):
                with self.subTest(config=config, weights=weights):
                    (target / "config.json").write_text(config)
                    (target / "model.safetensors").write_text(weights)
                    self.assertFalse(ModelStore._has_ready_model(target, spec))

    def test_nested_codec_shards_and_unsafe_weight_index_are_rejected(self):
        spec = get_model_spec("qwen3_tts")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model"
            write_snapshot(target, spec)
            nested = target / "speech_tokenizer"
            (nested / "model.safetensors").unlink()
            (nested / "model-00001-of-00002.safetensors").write_text("weights")
            self.assertFalse(ModelStore._has_ready_model(target, spec))
            (nested / "model-00002-of-00002.safetensors").write_text("weights")
            self.assertTrue(ModelStore._has_ready_model(target, spec))
            (target.parent / "outside.safetensors").write_text("weights")
            (target / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"tensor": "../outside.safetensors"},
            }))
            self.assertFalse(ModelStore._has_ready_model(target, spec))

    def test_relocated_legacy_manifest_uses_current_root_and_survives_original_removal(self):
        spec = get_model_spec("kokoro")
        downloader = Mock(side_effect=AssertionError("no network expected"))
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary).resolve() / "original"
            store = self.make_store(original)
            target = store._target_dir(spec.key, spec.repo_candidates[0])
            write_snapshot(target, spec)
            (original / "resolved_models.json").write_text(json.dumps({"version": 1, "models": {
                spec.key: {"repo_id": spec.repo_candidates[0], "path": str(target)},
            }}))
            copied = original.parent / "copied"
            shutil.copytree(original, copied)
            migrated = self.make_store(copied, downloader)
            result = migrated.ensure_model(spec, offline=True)
            self.assertTrue(result.is_relative_to(copied))
            entry = migrated._load_manifest()["models"][spec.key]
            self.assertEqual(copied / entry["path"], result)
            self.assertIsNone(entry["revision"])
            self.assertEqual(entry["validation"], "local")
            shutil.rmtree(original)
            self.assertEqual(migrated.ensure_model(spec, offline=True), result)
            downloader.assert_not_called()

    def test_external_manifest_and_symlink_paths_do_not_use_original_models(self):
        spec = get_model_spec("dia")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outside = root / "outside"
            write_snapshot(outside, spec)
            store = self.make_store(root / "active")
            for stored_path in (str(outside), "../outside"):
                store.manifest_path.write_text(json.dumps({"version": 1, "models": {
                    spec.key: {"repo_id": spec.repo_candidates[0], "path": stored_path},
                }}))
                with self.assertRaisesRegex(RuntimeError, "offline mode"):
                    store.ensure_model(spec, offline=True)
            canonical = store._target_dir(spec.key, spec.repo_candidates[0])
            canonical.parent.mkdir(parents=True)
            canonical.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "escapes the active root"):
                store.ensure_model(spec, offline=True)
            self.assertEqual((outside / "config.json").read_text(), "{}")

    def test_external_manifest_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outside = root / "manifest.json"
            outside.write_text('{"version":1,"models":{}}')
            store = self.make_store(root / "active")
            store.manifest_path.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "escapes the active root"):
                store.ensure_model(get_model_spec("dia"), offline=True)

    def test_malformed_manifest_is_preserved_and_never_silently_overwritten(self):
        downloader = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary), downloader)
            payloads = ["{", "[]", '{"models":[]}', '{"version":99,"models":{}}',
                        '{"models":{"dia":[]}}',
                        '{"models":{"dia":{"repo_id":"repo","path":"path","complete":"yes"}}}',
                        '{"models":{"dia":{"repo_id":"repo","path":"path","revision":[]}}}']
            for content in payloads:
                with self.subTest(content=content):
                    store.manifest_path.write_text(content)
                    with self.assertRaisesRegex(RuntimeError, "Cannot read model manifest"):
                        store.ensure_model(get_model_spec("dia"))
                    self.assertEqual(store.manifest_path.read_text(), content)
            downloader.assert_not_called()

    def test_completion_inventory_detects_removed_or_resized_files(self):
        spec = get_model_spec("kokoro")
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary))
            target = store._target_dir(spec.key, spec.repo_candidates[0])
            write_snapshot(target, spec)
            store.ensure_model(spec, offline=True)
            (target / "model.safetensors").write_text("short")
            with self.assertRaisesRegex(RuntimeError, "offline mode"):
                store.ensure_model(spec, offline=True)

    def test_malformed_completion_marker_cannot_be_adopted_as_legacy(self):
        spec = get_model_spec("dia")
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary))
            target = store._target_dir(spec.key, spec.repo_candidates[0])
            write_snapshot(target, spec)
            store.ensure_model(spec, offline=True)
            marker = target / COMPLETION_FILE
            original = json.loads(marker.read_text())
            corruptions = ["{", json.dumps({**original, "files": {"../outside": 1}}),
                           json.dumps({**original, "required_assets": None}),
                           json.dumps({**original, "required_assets": ["../outside"]})]
            for content in corruptions:
                marker.write_text(content)
                with self.assertRaisesRegex(RuntimeError, "offline mode"):
                    store.ensure_model(spec, offline=True)

    def test_known_legacy_revision_is_recorded_without_remote_lookup(self):
        spec = get_model_spec("dia")
        loader = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary), repo_info_loader=loader)
            target = store._target_dir(spec.key, spec.repo_candidates[0])
            write_snapshot(target, spec)
            metadata = target / ".cache/huggingface/download/model.safetensors.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(f"{REVISION}\netag\n1234\n")
            metadata.with_name("config.json.metadata").write_text(f"{REVISION}\netag\n1234\n")
            store.ensure_model(spec, offline=True)
            self.assertEqual(store._load_manifest()["models"][spec.key]["revision"], REVISION)
            loader.assert_not_called()

    def test_conversion_uses_staged_local_input_and_preserves_full_precision_policy(self):
        spec = get_model_spec("dia")

        def download(repo_id, local_dir, revision):
            target = Path(local_dir)
            (target / "config.json").write_text("{}")
            (target / "weights.pth").write_text("source weights")

        def convert(**kwargs):
            self.assertNotEqual(kwargs["hf_path"], kwargs["mlx_path"])
            self.assertTrue((Path(kwargs["hf_path"]) / "weights.pth").is_file())
            self.assertFalse(kwargs["quantize"])
            self.assertFalse(kwargs["dequantize"])
            shutil.copytree(kwargs["hf_path"], kwargs["mlx_path"], dirs_exist_ok=True)
            (Path(kwargs["mlx_path"]) / "model.safetensors").write_text("converted")

        converter = Mock(side_effect=convert)
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary), download, converter=converter)
            result = store.ensure_model(spec)
            self.assertTrue((result / "model.safetensors").is_file())
            self.assertEqual(json.loads((result / COMPLETION_FILE).read_text())["validation"], "converted")
            converter.assert_called_once()

    def test_inherited_offline_policy_prevents_acquisition_without_explicit_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            loader, downloader = Mock(), Mock()
            store = self.make_store(Path(temporary), downloader, repo_info_loader=loader)
            with patch("libro_tts.store.constants.HF_HUB_OFFLINE", True):
                with self.assertRaisesRegex(RuntimeError, "offline mode"):
                    store.ensure_model(get_model_spec("dia"))
            loader.assert_not_called()
            downloader.assert_not_called()

    def test_legacy_revision_is_unknown_when_primary_provenance_is_missing_or_mixed(self):
        spec = get_model_spec("dia")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            write_snapshot(target, spec)
            metadata = target / ".cache/huggingface/download/model.safetensors.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(f"{REVISION}\netag\n1234\n")
            self.assertIsNone(ModelStore._local_revision(target, spec))
            metadata.with_name("config.json.metadata").write_text(f"{'b' * 40}\netag\n1234\n")
            self.assertIsNone(ModelStore._local_revision(target, spec))

    def test_remote_inventory_cannot_escape_the_snapshot_or_replace_completion_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            downloader = Mock()
            for files in ({"../outside": 1}, {COMPLETION_FILE: 1}):
                with self.subTest(files=files):
                    store = self.make_store(Path(temporary), downloader,
                                            repo_info_loader=Mock(return_value=repo_info(files)))
                    with self.assertRaisesRegex(RuntimeError, "Invalid resumable download metadata"):
                        store.ensure_model(get_model_spec("dia"))
            downloader.assert_not_called()

    def test_config_replace_failure_preserves_existing_config(self):
        spec = get_model_spec("dia")
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary))
            target = store._target_dir(spec.key, spec.repo_candidates[0])
            write_snapshot(target, spec)
            with patch("libro_tts.store.os.replace", side_effect=OSError("simulated write failure")):
                with self.assertRaisesRegex(RuntimeError, "Cannot normalize model metadata"):
                    store.ensure_model(spec, offline=True)
            self.assertEqual((target / "config.json").read_text(), "{}")
            self.assertFalse((target / COMPLETION_FILE).exists())
            self.assertEqual(list(target.glob(".config.json.*.tmp")), [])

    def test_manifest_replace_failure_preserves_previous_json_and_cleans_temporary(self):
        spec = get_model_spec("dia")
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(Path(temporary))
            target = store._target_dir(spec.key, spec.repo_candidates[0])
            write_snapshot(target, spec)
            original = '{"version":1,"models":{}}'
            store.manifest_path.write_text(original)
            from libro_tts import store as store_module
            real_replace = store_module.os.replace

            def fail_manifest(source, destination):
                if Path(destination) == store.manifest_path:
                    raise OSError("simulated publication failure")
                return real_replace(source, destination)

            with patch.object(store_module.os, "replace", side_effect=fail_manifest):
                with self.assertRaisesRegex(RuntimeError, "Cannot update model manifest"):
                    store.ensure_model(spec, offline=True)
            self.assertEqual(store.manifest_path.read_text(), original)
            self.assertEqual(list(store.root_dir.glob(".resolved_models.json.*.tmp")), [])
            self.assertEqual(store.ensure_model(spec, offline=True), target)

    def test_invalid_catalog_paths_are_rejected_before_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            downloader = Mock()
            store = self.make_store(Path(temporary), downloader)
            for spec in (replace(get_model_spec("dia"), key="../escape"),
                         replace(get_model_spec("dia"), repo_candidates=("mlx-community/../escape",))):
                with self.assertRaisesRegex(RuntimeError, "Invalid"):
                    store.ensure_model(spec)
            downloader.assert_not_called()

    def test_different_model_processes_preserve_both_manifest_entries(self):
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for key in ("dia", "csm"):
                spec = get_model_spec(key)
                write_snapshot(root / key / spec.repo_candidates[0].replace("/", "__"), spec)
            start = context.Event()
            results = context.Queue()
            processes = [context.Process(target=concurrent_writer, args=(str(root), key, start, results))
                         for key in ("dia", "csm")]
            try:
                for process in processes:
                    process.start()
                start.set()
                outcomes = [results.get(timeout=15) for _ in processes]
                self.assertEqual(sorted(outcomes), [("csm", None), ("dia", None)])
                for process in processes:
                    process.join(timeout=5)
                    self.assertEqual(process.exitcode, 0)
                manifest = ModelStore(root_dir=root)._load_manifest()
                self.assertEqual(set(manifest["models"]), {"dia", "csm"})
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                results.close()


if __name__ == "__main__":
    unittest.main()
