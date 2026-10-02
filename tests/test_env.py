import os
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from pathlib import Path

from libro_tts.catalog import list_model_keys
from libro_tts import env


class EnvTests(unittest.TestCase):
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
        mock_run.return_value = MagicMock(returncode=0, stdout="Device(type=gpu, index=0)\n", stderr="")
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
