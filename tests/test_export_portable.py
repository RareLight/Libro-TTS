from contextlib import redirect_stdout
from contextlib import ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

from libro_tts.paths import project_root


spec = importlib.util.spec_from_file_location("portable_export", project_root() / "scripts/export_portable_build.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class PortableExportTests(unittest.TestCase):
    def make_project(self, root):
        root.mkdir()
        for name in exporter.COPY_PATHS:
            target = root / name
            if name in {"libro_tts", "models", "reference_voices", "docs"}:
                target.mkdir(parents=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("source fixture")
        (root / "README.md").write_text("Source readme")
        return root

    def run_export(self, root, destination):
        with ExitStack() as stack:
            stack.enter_context(patch.object(exporter, "PROJECT_ROOT", root))
            stack.enter_context(patch.object(exporter, "_build_requirements_text", return_value="locked requirements"))
            stack.enter_context(patch.object(exporter.sys, "prefix", str(root / ".venv")))
            stack.enter_context(patch.object(exporter.sys, "base_prefix", "/base"))
            stack.enter_context(patch.object(exporter.sys, "argv", ["export", "--output-dir", str(destination)]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return exporter.main()

    def test_export_carries_lock_setup_and_assets_but_excludes_credentials_and_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "source"
            root.mkdir()
            for name in exporter.COPY_PATHS:
                target = root / name
                if name in {"libro_tts", "models", "reference_voices"}:
                    target.mkdir(parents=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("locked source fixture")
            (root / "README.md").write_text("Source readme")
            hf = root / "models/.hf"
            hf.mkdir()
            for name in ("token", "stored_tokens", "partial.incomplete", "weights.safetensors"):
                (hf / name).write_text("fixture")
            (root / ".venv").mkdir()
            (root / ".venv/do-not-export").touch()
            destination = root.parent / "export"
            with patch.object(exporter, "PROJECT_ROOT", root), patch.object(
                exporter, "_build_requirements_text", return_value="locked requirements"
            ), patch.object(exporter.sys, "prefix", str(root / ".venv")), patch.object(
                exporter.sys, "base_prefix", "/base"
            ), patch.object(exporter.sys, "argv", ["export", "--output-dir", str(destination)]), redirect_stdout(io.StringIO()):
                self.assertEqual(exporter.main(), 0)

            self.assertEqual((destination / "uv.lock").read_text(), "locked source fixture")
            self.assertTrue((destination / "scripts/setup.sh").is_file())
            self.assertEqual(sorted(p.name for p in (destination / "models/.hf").iterdir()), ["weights.safetensors"])
            self.assertFalse((destination / ".venv").exists())
            self.assertIn("bash scripts/setup.sh --no-dev", (destination / "README.md").read_text())

    def test_internal_blob_links_survive_relocation_and_original_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = self.make_project(base / "source")
            blob = root / "models/.hf/hub/blobs/weight"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"tensor")
            links = root / "models/.hf/hub/snapshot"
            links.mkdir()
            (links / "relative.safetensors").symlink_to(os.path.relpath(blob, links))
            (links / "absolute.safetensors").symlink_to(blob)
            destination = base / "export"
            self.assertEqual(self.run_export(root, destination), 0)
            moved = base / "relocated with spaces"
            destination.rename(moved)
            shutil.rmtree(root)
            for name in ("relative.safetensors", "absolute.safetensors"):
                path = moved / "models/.hf/hub/snapshot" / name
                self.assertTrue(path.is_symlink())
                self.assertFalse(Path(os.readlink(path)).is_absolute())
                self.assertEqual(path.read_bytes(), b"tensor")

    def test_external_broken_and_excluded_target_links_fail_before_publication(self):
        for kind in ("external", "external-directory", "broken", "metadata", "credential", "xet", "cyclic"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary).resolve()
                root = self.make_project(base / "source")
                target = base / "private"
                if kind == "metadata":
                    target = root / "models/.cache/private"
                    target.parent.mkdir()
                if kind == "credential":
                    target = root / "models/token"
                if kind == "xet":
                    target = root / "models/.hf/xet/data"
                    target.parent.mkdir(parents=True)
                if kind == "external-directory":
                    target.mkdir()
                elif kind == "cyclic":
                    target = root / "models/config.json"
                elif kind != "broken":
                    target.write_text("private data")
                (root / "models/config.json").symlink_to(target)
                destination = base / "export"
                with self.assertRaises(RuntimeError):
                    self.run_export(root, destination)
                self.assertFalse(destination.exists())
                self.assertFalse(list(base.glob('.export.tmp-*')))

    def test_auxiliary_absolute_paths_relocated_and_existing_destinations_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = self.make_project(base / 'source')
            relative = Path('.hf/hub/models--example--codec/snapshots') / ('b' * 40)
            (root / 'models' / relative).mkdir(parents=True)
            (root / 'models/resolved_models.json').write_text(json.dumps({
                'version': 2, 'models': {}, 'assets': {
                    'codec': {'repo_id': 'example/codec', 'revision': 'b' * 40,
                              'path': '/old/checkout/models/' + relative.as_posix()},
                    'missing': {'repo_id': 'example/missing', 'revision': 'c' * 40,
                                'path': '/old/checkout/missing'},
                }}))
            destination = base / 'export'
            destination.mkdir()
            sentinel = destination / 'keep'
            sentinel.write_text('existing data')
            with self.assertRaisesRegex(SystemExit, 'already exists'):
                self.run_export(root, destination)
            self.assertEqual(sentinel.read_text(), 'existing data')
            shutil.rmtree(destination)
            destination.symlink_to(base / 'absent')
            with self.assertRaisesRegex(SystemExit, 'already exists'):
                self.run_export(root, destination)
            self.assertTrue(destination.is_symlink())
            destination.unlink()
            self.assertEqual(self.run_export(root, destination), 0)
            manifest = json.loads((destination / 'models/resolved_models.json').read_text())
            self.assertEqual(set(manifest['assets']), {'codec'})
            self.assertEqual(manifest['assets']['codec']['path'], relative.as_posix())

    def test_recursive_destination_is_rejected_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_project(Path(temporary).resolve() / "source")
            destination = root / "models/export"
            with self.assertRaisesRegex(SystemExit, "inside an included source"):
                self.run_export(root, destination)
            self.assertFalse(destination.exists())

    def test_source_only_export_allows_absent_optional_assets_and_copy_failure_cleans_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = self.make_project(base / "source")
            shutil.rmtree(root / "models")
            shutil.rmtree(root / "reference_voices")
            destination = base / "export"
            with patch.object(exporter, "_copy_path", side_effect=OSError("copy failed")):
                with self.assertRaisesRegex(OSError, "copy failed"):
                    self.run_export(root, destination)
            self.assertFalse(destination.exists())
            self.assertFalse(list(base.glob('.export.tmp-*')))
            self.assertEqual(self.run_export(root, destination), 0)
            self.assertTrue((destination / "uv.lock").is_file())
            self.assertFalse((destination / "models").exists())
            self.assertFalse((destination / "reference_voices").exists())

    def test_legacy_manifest_and_revision_adopted_only_in_export_before_metadata_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = self.make_project(base / "source")
            model = root / "models/kokoro/mlx-community__Kokoro-82M-bf16"
            (model / "voices").mkdir(parents=True)
            (model / "config.json").write_text('{"model_type":"kokoro"}')
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "voices/af_aoede.safetensors").write_bytes(b"voice")
            revision = 'a' * 40
            for name in ("config.json", "model.safetensors", "voices/af_aoede.safetensors"):
                metadata = model / ".cache/huggingface/download" / f'{name}.metadata'
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text(revision + '\netag\n0\n')
            manifest = root / "models/resolved_models.json"
            original = json.dumps({"version":1,"models":{
                "kokoro":{"repo_id":"mlx-community/Kokoro-82M-bf16","path":str(model)},
                "dia":{"repo_id":"mlx-community/Dia-1.6B-fp16","path":"/old/models/dia"}}})
            manifest.write_text(original)
            xet = root / "models/.hf/xet/logs"
            xet.mkdir(parents=True)
            (xet / 'machine.log').write_text('private diagnostics')
            destination = base / "export"
            self.assertEqual(self.run_export(root, destination), 0)
            result = json.loads((destination / "models/resolved_models.json").read_text())
            self.assertEqual(result['version'], 2)
            self.assertEqual(set(result['models']), {'kokoro'})
            self.assertEqual(result['models']['kokoro']['path'], 'kokoro/mlx-community__Kokoro-82M-bf16')
            self.assertEqual(result['models']['kokoro']['revision'], revision)
            self.assertEqual(manifest.read_text(), original)
            self.assertFalse((model / '.libro-model.json').exists())
            self.assertTrue((destination / "uv.lock").exists())
            self.assertFalse(list((destination / 'models').rglob('.cache')))
            self.assertFalse(list((destination / 'models').rglob('*.lock')))
            self.assertFalse((destination / 'models/.hf/xet').exists())


if __name__ == "__main__":
    unittest.main()
