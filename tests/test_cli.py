import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from libro_tts.cli import main
from libro_tts.runtime import BatchFailure, BatchRuntimeResult


class CliTests(unittest.TestCase):
    def test_diagnostics_exit_nonzero_on_required_probe_failure(self):
        from libro_tts.env import EnvironmentReport, TTSRuntimeProbe

        report = EnvironmentReport(
            python_executable="/project/.venv/bin/python", python_version="3.12.2",
            conda_env=None, expected_conda_env=None, package_versions={"mlx-audio": "0.4.3"},
            tts_runtime_probe=TTSRuntimeProbe("0.4.3", probe_error="probe failed"),
            is_project_virtualenv=True, is_local_ffmpeg=True,
        )
        with patch("libro_tts.cli.doctor", return_value=report), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--diag"]), 1)

    def test_diag_command(self):
        with patch("libro_tts.cli.doctor") as mock_doctor:
            mock_doctor.return_value = type(
                "Report",
                (),
                {
                    "python_executable": "/usr/bin/python",
                    "python_version": "3.11",
                    "conda_env": "libro",
                    "expected_conda_env": None,
                    "python_prefix": "/project/.venv",
                    "project_virtualenv": "/project/.venv",
                    "is_project_virtualenv": True,
                    "platform_system": "Darwin",
                    "platform_machine": "arm64",
                    "ffmpeg_executable": "/project/.venv/bin/ffmpeg",
                    "is_local_ffmpeg": True,
                    "package_versions": {
                        "mlx-audio": "0.3.1",
                        "mlx": "0.30.6",
                        "huggingface_hub": "1.4.1",
                    },
                    "tts_runtime_probe": type(
                        "Probe",
                        (),
                        {
                            "mlx_audio_version": "0.4.3",
                            "available_model_types": ("kokoro", "voxtral_tts"),
                            "model_remapping": {"csm": "sesame"},
                            "probe_error": None,
                        },
                    )(),
                    "tts_model_support": {
                        "kokoro": "supported",
                        "voxtral_tts": "supported",
                    },
                },
            )()
            out = io.StringIO()
            with redirect_stdout(out):
                rc = main(["--diag"])

        self.assertEqual(rc, 0)
        value = out.getvalue()
        self.assertIn("Libro-TTS diagnostics", value)
        self.assertIn("tts_model_support:", value)
        self.assertIn("voxtral_tts: supported", value)

    def test_list_models_command(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--list-models"])

        self.assertEqual(rc, 0)
        value = out.getvalue()
        self.assertIn("Supported models:", value)
        self.assertIn("- voxtral_tts: Voxtral 4B TTS", value)

    def test_list_kokoro_voices_command(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--list-kokoro-voices"])

        self.assertEqual(rc, 0)
        value = out.getvalue()
        self.assertIn("Kokoro English voices:", value)
        self.assertIn("af_aoede", value)
        self.assertIn("bm_lewis", value)

    @patch("libro_tts.runtime.process_single_file")
    @patch("libro_tts.cli.configure_logging")
    @patch("libro_tts.cli.configure_local_cache_environment")
    @patch("libro_tts.cli.prepare_runtime_dirs")
    @patch("libro_tts.cli.validate_tts_model_runtime_support")
    @patch("libro_tts.cli.validate_mlx_backend_preflight")
    @patch("libro_tts.cli.validate_runtime_environment")
    @patch("libro_tts.store.ModelStore")
    def test_generation_runs_mlx_preflight(
        self,
        mock_model_store,
        mock_validate_env,
        mock_preflight,
        mock_validate_model_support,
        mock_prepare_dirs,
        mock_configure_cache,
        mock_configure_logging,
        mock_process_single_file,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_file = tmp / "input.txt"
            input_file.write_text("hello", encoding="utf-8")

            mock_prepare_dirs.return_value = {
                "models_dir": tmp / "models",
                "hf_cache_dir": tmp / "models" / ".hf",
            }
            mock_configure_logging.return_value = __import__("logging").getLogger("test")

            rc = main([str(input_file), "--output", str(tmp / "out.wav")])

        self.assertEqual(rc, 0)
        self.assertTrue(mock_validate_env.called)
        self.assertTrue(mock_preflight.called)
        self.assertTrue(mock_validate_model_support.called)
        self.assertTrue(mock_model_store.called)
        self.assertTrue(mock_process_single_file.called)
        self.assertTrue(mock_configure_cache.called)

    @patch("libro_tts.runtime.process_batch_dir")
    @patch("libro_tts.cli.configure_logging")
    @patch("libro_tts.cli.configure_local_cache_environment")
    @patch("libro_tts.cli.prepare_runtime_dirs")
    @patch("libro_tts.cli.validate_tts_model_runtime_support")
    @patch("libro_tts.cli.validate_mlx_backend_preflight")
    @patch("libro_tts.cli.validate_runtime_environment")
    @patch("libro_tts.store.ModelStore")
    def test_batch_returns_nonzero_when_any_file_fails(
        self,
        mock_model_store,
        _mock_validate_env,
        _mock_preflight,
        _mock_validate_model_support,
        mock_prepare_dirs,
        _mock_configure_cache,
        _mock_configure_logging,
        mock_process_batch_dir,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_dir = tmp / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "a.txt").write_text("ok", encoding="utf-8")

            mock_prepare_dirs.return_value = {
                "models_dir": tmp / "models",
                "hf_cache_dir": tmp / "models" / ".hf",
            }
            mock_process_batch_dir.return_value = BatchRuntimeResult(
                total_files=2,
                files_processed=1,
                files_failed=1,
                failures=[BatchFailure(input_file="b.txt", error="decode failed")],
            )

            out = io.StringIO()
            with redirect_stdout(out):
                rc = main(
                    [
                        "--input-dir",
                        str(input_dir),
                    ]
                )

        self.assertEqual(rc, 1)
        self.assertIn("Batch completed with failures", out.getvalue())
        self.assertTrue(mock_model_store.called)

    @patch("libro_tts.runtime.process_single_file")
    @patch("libro_tts.cli.configure_logging")
    @patch("libro_tts.cli.configure_local_cache_environment")
    @patch("libro_tts.cli.prepare_runtime_dirs")
    @patch("libro_tts.cli.validate_tts_model_runtime_support")
    @patch("libro_tts.cli.validate_mlx_backend_preflight")
    @patch("libro_tts.cli.validate_runtime_environment")
    @patch("libro_tts.store.ModelStore")
    def test_single_default_output_keeps_audio_extension_for_dotted_stems(
        self,
        _mock_model_store,
        _mock_validate_env,
        _mock_preflight,
        _mock_validate_model_support,
        mock_prepare_dirs,
        _mock_configure_cache,
        mock_configure_logging,
        mock_process_single_file,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_file = tmp / "06 - Chapter 1. When Was Canada_.txt"
            input_file.write_text("hello", encoding="utf-8")
            mock_prepare_dirs.return_value = {
                "models_dir": tmp / "models",
                "hf_cache_dir": tmp / "models" / ".hf",
            }
            mock_configure_logging.return_value = __import__("logging").getLogger("test")

            rc = main([str(input_file)])

        self.assertEqual(rc, 0)
        self.assertTrue(mock_process_single_file.called)
        called_output_prefix = mock_process_single_file.call_args.kwargs["output_prefix"]
        self.assertTrue(called_output_prefix.endswith(".wav"))

    @patch("libro_tts.runtime.process_single_file")
    @patch("libro_tts.cli.configure_logging")
    @patch("libro_tts.cli.configure_local_cache_environment")
    @patch("libro_tts.cli.prepare_runtime_dirs")
    @patch("libro_tts.cli.validate_tts_model_runtime_support")
    @patch("libro_tts.cli.validate_mlx_backend_preflight")
    @patch("libro_tts.cli.validate_runtime_environment")
    @patch("libro_tts.store.ModelStore")
    def test_single_default_output_avoids_double_extension_for_audio_like_stem(
        self,
        _mock_model_store,
        _mock_validate_env,
        _mock_preflight,
        _mock_validate_model_support,
        mock_prepare_dirs,
        _mock_configure_cache,
        mock_configure_logging,
        mock_process_single_file,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_file = tmp / "foo.wav.txt"
            input_file.write_text("hello", encoding="utf-8")
            mock_prepare_dirs.return_value = {
                "models_dir": tmp / "models",
                "hf_cache_dir": tmp / "models" / ".hf",
            }
            mock_configure_logging.return_value = __import__("logging").getLogger("test")

            rc = main([str(input_file)])

        self.assertEqual(rc, 0)
        called_output_prefix = mock_process_single_file.call_args.kwargs["output_prefix"]
        self.assertTrue(called_output_prefix.endswith("foo.wav"))

    @patch("libro_tts.runtime.process_single_file")
    @patch("libro_tts.cli.configure_logging")
    @patch("libro_tts.cli.configure_local_cache_environment")
    @patch("libro_tts.cli.prepare_runtime_dirs")
    @patch("libro_tts.cli.validate_tts_model_runtime_support")
    @patch("libro_tts.cli.validate_mlx_backend_preflight")
    @patch("libro_tts.cli.validate_runtime_environment")
    @patch("libro_tts.store.ModelStore")
    def test_cli_model_alias_resolves_to_canonical_model_key(
        self,
        _mock_model_store,
        _mock_validate_env,
        _mock_preflight,
        _mock_validate_model_support,
        mock_prepare_dirs,
        _mock_configure_cache,
        mock_configure_logging,
        mock_process_single_file,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_file = tmp / "input.txt"
            input_file.write_text("hello", encoding="utf-8")
            mock_prepare_dirs.return_value = {
                "models_dir": tmp / "models",
                "hf_cache_dir": tmp / "models" / ".hf",
            }
            mock_configure_logging.return_value = __import__("logging").getLogger("test")

            rc = main([str(input_file), "--model", "spark-tts"])

        self.assertEqual(rc, 0)
        options = mock_process_single_file.call_args.kwargs["options"]
        self.assertEqual(options.model_key, "spark")


if __name__ == "__main__":
    unittest.main()
