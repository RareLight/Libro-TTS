import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from libro_tts.paths import project_root


class LauncherTests(unittest.TestCase):
    def test_missing_environment_is_actionable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(project_root() / "run.sh", root / "run.sh")
            result = subprocess.run(["bash", str(root / "run.sh")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("scripts/setup.sh", result.stderr)

    def test_symlink_launch_preserves_arguments_cwd_and_clears_python_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "project with spaces"
            root.mkdir()
            shutil.copy2(project_root() / "run.sh", root / "run.sh")
            local_bin = root / ".venv/bin"
            local_bin.mkdir(parents=True)
            interpreter = local_bin / "python"
            interpreter.write_text(
                '#!/usr/bin/env bash\n'
                'printf "%s\\n" "$PWD" "${PYTHONPATH-unset}" "${PYTHONHOME-unset}" "$@"\n'
            )
            interpreter.chmod(0o755)
            caller = root.parent / "caller"
            caller.mkdir()
            alias = caller / "libro-tts"
            alias.symlink_to(root / "run.sh")
            environment = dict(
                os.environ, PYTHONPATH="/unrelated/packages", PYTHONHOME="/wrong/python",
                CONDA_DEFAULT_ENV="unrelated", VIRTUAL_ENV="/unrelated/venv",
            )
            result = subprocess.run(
                ["bash", str(alias), "input with spaces.txt", "-o", "output with spaces.wav"],
                cwd=caller, env=environment, text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout.splitlines(), [
                str(caller), "unset", "unset", str(root / "Libro-tts.py"),
                "input with spaces.txt", "-o", "output with spaces.wav",
            ])


if __name__ == "__main__":
    unittest.main()
