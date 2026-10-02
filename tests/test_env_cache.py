import os
import tempfile
import unittest
from pathlib import Path

from libro_tts.env import configure_local_cache_environment


class EnvCacheTests(unittest.TestCase):
    def test_configure_local_cache_environment_sets_roots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "hf"
            configure_local_cache_environment(root)

            self.assertEqual(os.environ["HF_HOME"], str(root))
            self.assertTrue((root / "hub").exists())
            self.assertTrue((root / "transformers").exists())
            self.assertTrue((root / "datasets").exists())


if __name__ == "__main__":
    unittest.main()
