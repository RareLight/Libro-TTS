from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libro_tts.paths import project_root


spec = importlib.util.spec_from_file_location("portable_export", project_root() / "scripts/export_portable_build.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class PortableExportTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
