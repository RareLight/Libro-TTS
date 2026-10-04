import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from libro_tts.bootstrap import configure_local_cache_environment
from libro_tts.paths import hf_cache_dir, project_root


class BootstrapTests(unittest.TestCase):
    def run_fresh(self, script, **environment):
        child_env = dict(os.environ)
        child_env.update(environment)
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=project_root(),
            env=child_env, text=True, capture_output=True, check=True,
        )
        return json.loads(result.stdout.splitlines()[-1])

    def test_package_and_cli_import_do_not_import_runtime_dependencies(self):
        modules = self.run_fresh(
            "import sys, json; import libro_tts.cli; "
            "print(json.dumps([name for name in "
            "('numpy', 'huggingface_hub', 'mlx_audio') if name in sys.modules]))"
        )
        self.assertEqual(modules, [])

    def test_fresh_dependency_constants_override_inherited_global_caches(self):
        values = self.run_fresh(
            "import json; import libro_tts; from huggingface_hub import constants; "
            "print(json.dumps([constants.HF_HOME, constants.HF_HUB_CACHE, "
            "constants.HF_XET_CACHE, constants.HF_ASSETS_CACHE]))",
            HF_HOME="/global/hf", HF_HUB_CACHE="/global/hub",
            HF_XET_CACHE="/global/xet", HF_ASSETS_CACHE="/global/assets",
        )
        root = hf_cache_dir()
        self.assertEqual(values, [str(root), str(root / "hub"), str(root / "xet"), str(root / "assets")])

    def test_offline_policy_precedes_dependency_import_in_fresh_process(self):
        value = self.run_fresh(
            "import json; from libro_tts.bootstrap import configure_local_cache_environment; "
            "configure_local_cache_environment(offline=True); "
            "from huggingface_hub import constants; print(json.dumps(constants.HF_HUB_OFFLINE))",
            HF_HUB_OFFLINE="0",
        )
        self.assertTrue(value)

    def test_benchmark_rejects_wrong_environment_before_runtime_import(self):
        result = self.run_fresh(
            "import json, runpy, sys; "
            "namespace = runpy.run_path('scripts/benchmark_chunk_parallel.py'); "
            "sys.prefix = '/wrong/environment'; "
            "sys.argv = ['benchmark', '--input-file', 'unused.txt']; "
            "status = namespace['main'](); "
            "print(json.dumps([status, 'libro_tts.runtime' in sys.modules, "
            "'mlx_audio' in sys.modules]))"
        )
        self.assertEqual(result, [1, False, False])

    def test_configuration_does_not_create_state_or_clear_inherited_offline(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            root = Path(temporary) / "not-created"
            configure_local_cache_environment(root)
            self.assertFalse(root.exists())
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")

    def test_path_bootstrap_is_idempotent(self):
        with patch.dict(os.environ):
            configure_local_cache_environment()
            configure_local_cache_environment()
            self.assertEqual(os.environ["PATH"].split(os.pathsep).count(str(project_root() / ".venv/bin")), 1)

    def test_preflight_child_inherits_offline_cache_policy_without_app_import(self):
        with patch.dict(os.environ):
            configure_local_cache_environment(offline=True)
            values = self.run_fresh(
                "import json; from huggingface_hub import constants; "
                "print(json.dumps([constants.HF_HUB_CACHE, constants.HF_HUB_OFFLINE]))"
            )
            self.assertEqual(values, [str(hf_cache_dir() / "hub"), True])


if __name__ == "__main__":
    unittest.main()
