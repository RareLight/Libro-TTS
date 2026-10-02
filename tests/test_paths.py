import unittest

from libro_tts.paths import hf_cache_dir, models_dir, prepare_runtime_dirs, project_root


class PathTests(unittest.TestCase):
    def test_project_structure_paths(self):
        root = project_root()
        self.assertTrue((root / "Libro-tts.py").exists())
        self.assertEqual(models_dir(), root / "models")
        self.assertEqual(hf_cache_dir(), root / "models" / ".hf")

    def test_prepare_runtime_dirs_creates_directories(self):
        dirs = prepare_runtime_dirs()
        self.assertTrue(dirs["models_dir"].exists())
        self.assertTrue(dirs["hf_cache_dir"].exists())


if __name__ == "__main__":
    unittest.main()
