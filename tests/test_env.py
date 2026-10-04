import os
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from pathlib import Path

from libro_tts.catalog import list_model_keys
from libro_tts import env
from libro_tts.paths import project_root


class EnvTests(unittest.TestCase):
    def setUp(self):
        prefix = patch.object(env.sys, "prefix", str(project_root() / ".venv"))
        base = patch.object(env.sys, "base_prefix", "/base-python")
        prefix.start()
        base.start()
        self.addCleanup(prefix.stop)
        self.addCleanup(base.stop)
        distribution = patch.object(env.importlib.metadata, "distribution")
        mocked_distribution = distribution.start()
        mocked_distribution.return_value.locate_file.return_value = project_root() / ".venv/lib/python3.12/site-packages"
        self.addCleanup(distribution.stop)

    def test_generation_rejects_external_environment_before_dependency_imports(self):
        with patch.object(env.sys, "prefix", "/unrelated/.venv"), patch.object(
            env.importlib, "import_module"
        ) as importer:
            with self.assertRaisesRegex(RuntimeError, "requires this project's .venv"):
                env.validate_runtime_environment()
            importer.assert_not_called()

    def test_project_environment_identity_does_not_resolve_python_symlink(self):
        with patch.object(env.sys, "executable", "/base-python/bin/python"):
            self.assertTrue(env.is_project_virtualenv())

    def test_base_environment_cannot_impersonate_project_virtualenv(self):
        with patch.object(env.sys, "base_prefix", env.sys.prefix):
            self.assertFalse(env.is_project_virtualenv())

    @patch("libro_tts.env._collect_package_versions", return_value={"mlx-audio": "0.4.3"})
    @patch("libro_tts.env.importlib.import_module")
    def test_dependency_metadata_outside_local_environment_is_rejected(self, _imports, _versions):
        with patch.object(env.importlib.metadata, "distribution") as distribution:
            distribution.return_value.locate_file.return_value = "/shared/site-packages"
            with self.assertRaisesRegex(RuntimeError, "outside the project environment"):
                env.validate_runtime_environment()

    def test_external_encoder_is_rejected(self):
        with patch.object(env.shutil, "which", return_value="/opt/homebrew/bin/ffmpeg"):
            with self.assertRaisesRegex(RuntimeError, "project-local FFmpeg"):
                env.validate_local_encoder()

    def test_missing_encoder_is_rejected(self):
        with patch.object(env.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "project-local FFmpeg"):
                env.validate_local_encoder()

    @patch("libro_tts.env.importlib.import_module")
    @patch("libro_tts.env._collect_package_versions")
    def test_validate_runtime_environment_passes_without_env_name_requirement(
        self,
        mock_versions,
        mock_import,
    ):
        mock_versions.return_value = {
            "mlx-audio": "0.3.1",
            "mlx": "0.30.6",
            "huggingface_hub": "1.4.1",
        }
        with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "base"}, clear=False):
            report = env.validate_runtime_environment()

        self.assertEqual(report.conda_env, "base")
        self.assertIsNone(report.expected_conda_env)
        self.assertTrue(mock_import.called)

    @patch("libro_tts.env.importlib.import_module")
    @patch("libro_tts.env._collect_package_versions")
    def test_validate_runtime_environment_rejects_wrong_env_when_explicitly_configured(
        self,
        mock_versions,
        mock_import,
    ):
        mock_versions.return_value = {
            "mlx-audio": "0.3.1",
            "mlx": "0.30.6",
            "huggingface_hub": "1.4.1",
        }
        with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "base"}, clear=False):
            with self.assertRaises(RuntimeError):
                env.validate_runtime_environment(expected_conda_env="tts")

    @patch("libro_tts.env.importlib.import_module")
    @patch("libro_tts.env._collect_package_versions")
    def test_skip_env_check_allows_non_matching_explicit_env(self, mock_versions, mock_import):
        mock_versions.return_value = {
            "mlx-audio": "0.3.1",
            "mlx": "0.30.6",
            "huggingface_hub": "1.4.1",
        }
        with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "base"}, clear=False):
            report = env.validate_runtime_environment(
                expected_conda_env="tts",
                skip_env_check=True,
            )

        self.assertEqual(report.expected_conda_env, "tts")

    @patch("libro_tts.env.importlib.import_module")
    @patch("libro_tts.env._collect_package_versions")
    def test_env_variable_can_configure_expected_conda_env(self, mock_versions, mock_import):
        mock_versions.return_value = {
            "mlx-audio": "0.3.1",
            "mlx": "0.30.6",
            "huggingface_hub": "1.4.1",
        }
        with patch.dict(
            os.environ,
            {"CONDA_DEFAULT_ENV": "libro", "LIBRO_TTS_EXPECTED_CONDA_ENV": "libro"},
            clear=False,
        ):
            report = env.validate_runtime_environment()

        self.assertEqual(report.expected_conda_env, "libro")
        self.assertTrue(mock_import.called)

    @patch("libro_tts.env.subprocess.run")
    def test_validate_mlx_backend_preflight_passes(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout='{"available_model_types":["kokoro"],"probe_error":null}\n', stderr="")
        env.validate_mlx_backend_preflight(expected_conda_env="tts", skip_preflight=False)
        self.assertTrue(mock_run.called)

    @patch("libro_tts.env.subprocess.run")
    def test_validate_mlx_backend_preflight_raises_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=134,
            stdout="",
            stderr="Abort trap: 6",
        )
        with self.assertRaises(RuntimeError) as ctx:
            env.validate_mlx_backend_preflight(expected_conda_env="tts", skip_preflight=False)

        message = str(ctx.exception)
        self.assertIn("MLX backend failed preflight", message)
        self.assertIn("conda run -n tts python -c", message)

    @patch("libro_tts.env._collect_package_versions")
    @patch("libro_tts.env.subprocess.run")
    def test_probe_tts_runtime_support_parses_capabilities(self, mock_run, mock_versions):
        mock_versions.return_value = {"mlx-audio": "0.4.3"}
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                '{"mlx_audio_version":"0.4.3","available_model_types":["kokoro","qwen3_tts","sesame","spark","chatterbox","soprano","voxtral_tts","dia"],'
                '"model_remapping":{"csm":"sesame","voxtral_tts":"voxtral_tts"},"probe_error":null}'
            ),
            stderr="",
        )

        probe = env.probe_tts_runtime_support()

        self.assertEqual(probe.mlx_audio_version, "0.4.3")
        self.assertIn("voxtral_tts", probe.available_model_types)
        self.assertEqual(probe.model_remapping["csm"], "sesame")
        self.assertIsNone(probe.probe_error)

    @patch("libro_tts.env.probe_tts_runtime_support")
    def test_doctor_reports_all_catalog_models_supported_when_probe_matches(self, mock_probe):
        mock_probe.return_value = env.TTSRuntimeProbe(
            mlx_audio_version="0.4.3",
            available_model_types=(
                "chatterbox",
                "dia",
                "kokoro",
                "qwen3_tts",
                "sesame",
                "soprano",
                "spark",
                "voxtral_tts",
            ),
            model_remapping={"csm": "sesame", "voxtral_tts": "voxtral_tts"},
        )

        report = env.doctor(include_tts_runtime_probe=True)

        self.assertEqual(set(report.tts_model_support.keys()), set(list_model_keys()))
        self.assertTrue(all(status == "supported" for status in report.tts_model_support.values()))

    @patch("libro_tts.env.probe_tts_runtime_support")
    def test_validate_tts_model_runtime_support_raises_when_model_missing(self, mock_probe):
        mock_probe.return_value = env.TTSRuntimeProbe(
            mlx_audio_version="0.3.1",
            available_model_types=("kokoro", "qwen3_tts", "sesame", "spark"),
            model_remapping={"csm": "sesame"},
        )

        with self.assertRaises(RuntimeError) as ctx:
            env.validate_tts_model_runtime_support("voxtral_tts", expected_conda_env="tts")

        message = str(ctx.exception)
        self.assertIn("voxtral_tts", message)
        self.assertIn("0.3.1", message)
        self.assertIn("conda run -n tts python Libro-tts.py <args>", message)

    @patch("libro_tts.env.probe_tts_runtime_support")
    def test_validate_tts_model_runtime_support_checks_non_voxtral_models_too(self, mock_probe):
        mock_probe.return_value = env.TTSRuntimeProbe(
            mlx_audio_version="0.4.3",
            available_model_types=("kokoro", "qwen3_tts", "sesame", "voxtral_tts"),
            model_remapping={"csm": "sesame"},
        )

        with self.assertRaises(RuntimeError) as ctx:
            env.validate_tts_model_runtime_support("spark", expected_conda_env="tts")

        message = str(ctx.exception)
        self.assertIn("spark", message)
        self.assertIn("Expected upstream TTS model type: spark", message)

    def test_validate_model_load_preflight_ignores_non_voxtral_models(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            env.validate_model_load_preflight("kokoro", str(model_dir))

    def test_validate_model_load_preflight_raises_when_tekken_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            with self.assertRaises(RuntimeError) as ctx:
                env.validate_model_load_preflight("voxtral_tts", str(model_dir), expected_conda_env="tts")

        message = str(ctx.exception)
        self.assertIn("tekken.json", message)
        self.assertIn("conda run -n tts python Libro-tts.py <args>", message)

    @patch("libro_tts.env._collect_package_versions")
    @patch("libro_tts.env._get_mistral_tokenizer_cls")
    def test_validate_model_load_preflight_surfaces_mistral_common_schema_error(
        self,
        mock_tokenizer_cls,
        mock_versions,
    ):
        mock_tokenizer_cls.return_value.from_file.side_effect = TypeError(
            "AudioConfig.__init__() got an unexpected keyword argument 'voice_num_audio_tokens'"
        )
        mock_versions.return_value = {"mistral-common": "1.8.5"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "tekken.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx:
                env.validate_model_load_preflight(
                    "voxtral_tts",
                    str(model_dir),
                    expected_conda_env="tts",
                )

        message = str(ctx.exception)
        self.assertIn("mistral-common", message)
        self.assertIn("1.8.5", message)
        self.assertIn("voice_num_audio_tokens", message)


if __name__ == "__main__":
    unittest.main()
