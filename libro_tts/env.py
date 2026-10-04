from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .bootstrap import configure_local_cache_environment as configure_local_cache_environment
from .catalog import canonical_model_type_for_key, list_model_keys
from .paths import project_root

REQUIRED_PACKAGES = (
    "mlx-audio", "mlx", "huggingface_hub", "numpy", "spacy",
    "en-core-web-sm", "mistral-common", "imageio-ffmpeg",
)
MAX_PREFLIGHT_LOG_LINES = 40


@dataclass(frozen=True)
class TTSRuntimeProbe:
    mlx_audio_version: str | None
    available_model_types: tuple[str, ...] = ()
    model_remapping: dict[str, str] = field(default_factory=dict)
    probe_error: str | None = None


@dataclass(frozen=True)
class EnvironmentReport:
    python_executable: str
    python_version: str
    conda_env: str | None
    expected_conda_env: str | None
    package_versions: dict[str, str | None]
    tts_runtime_probe: TTSRuntimeProbe
    tts_model_support: dict[str, str] = field(default_factory=dict)
    python_prefix: str = ""
    project_virtualenv: str = ""
    is_project_virtualenv: bool = False
    platform_system: str = ""
    platform_machine: str = ""
    ffmpeg_executable: str | None = None
    is_local_ffmpeg: bool = False


def _collect_package_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _resolve_expected_conda_env(expected_conda_env: str | None = None) -> str | None:
    if expected_conda_env is not None:
        normalized = expected_conda_env.strip()
        return normalized or None

    configured = os.getenv("LIBRO_TTS_EXPECTED_CONDA_ENV", "").strip()
    return configured or None


def _runtime_command(expected_conda_env: str | None) -> str:
    if expected_conda_env:
        return f"conda run -n {expected_conda_env} python Libro-tts.py <args>"
    return "bash run.sh <args>"


def _mlx_import_check_command(expected_conda_env: str | None) -> str:
    if expected_conda_env:
        return (
            f"conda run -n {expected_conda_env} python -c "
            "\"from mlx_audio.tts.utils import load_model; print('mlx-audio runtime import OK')\""
        )
    return '.venv/bin/python -c "from mlx_audio.tts.utils import load_model; print(\'mlx-audio runtime import OK\')"'


def _trim_process_output(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) <= MAX_PREFLIGHT_LOG_LINES:
        return "\n".join(lines)
    trimmed = lines[:MAX_PREFLIGHT_LOG_LINES]
    trimmed.append(f"... ({len(lines) - MAX_PREFLIGHT_LOG_LINES} more lines omitted)")
    return "\n".join(trimmed)


def _empty_tts_runtime_probe(mlx_audio_version: str | None) -> TTSRuntimeProbe:
    return TTSRuntimeProbe(mlx_audio_version=mlx_audio_version)


def _tts_runtime_probe_script() -> str:
    return (
        "import importlib.metadata\n"
        "import json\n"
        "payload = {\n"
        "    'mlx_audio_version': None,\n"
        "    'available_model_types': [],\n"
        "    'model_remapping': {},\n"
        "    'probe_error': None,\n"
        "}\n"
        "try:\n"
        "    payload['mlx_audio_version'] = importlib.metadata.version('mlx-audio')\n"
        "except importlib.metadata.PackageNotFoundError:\n"
        "    pass\n"
        "try:\n"
        "    from mlx_audio.tts import utils as tts_utils\n"
        "    available = set(tts_utils.get_available_models())\n"
        "    available.discard('__pycache__')\n"
        "    payload['available_model_types'] = sorted(str(item) for item in available)\n"
        "    remapping = getattr(tts_utils, 'MODEL_REMAPPING', {})\n"
        "    if isinstance(remapping, dict):\n"
        "        payload['model_remapping'] = {str(key): str(value) for key, value in remapping.items()}\n"
        "except Exception as exc:\n"
        "    payload['probe_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )


def _get_mistral_tokenizer_cls():
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    return MistralTokenizer


def probe_tts_runtime_support(timeout_seconds: int = 20) -> TTSRuntimeProbe:
    fallback_version = _collect_package_versions(("mlx-audio",)).get("mlx-audio")
    command = [sys.executable, "-c", _tts_runtime_probe_script()]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(os.environ),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TTSRuntimeProbe(
            mlx_audio_version=fallback_version,
            probe_error=f"TTS runtime capability probe timed out after {timeout_seconds}s.",
        )

    stdout = result.stdout.strip()
    if result.returncode != 0:
        details = [f"TTS runtime capability probe exited with code {result.returncode}."]
        if stdout:
            details.append("stdout:")
            details.append(_trim_process_output(stdout))
        if result.stderr.strip():
            details.append("stderr:")
            details.append(_trim_process_output(result.stderr))
        return TTSRuntimeProbe(
            mlx_audio_version=fallback_version,
            probe_error="\n".join(details),
        )

    try:
        payload_line = next(
            (line for line in reversed(stdout.splitlines()) if line.strip()),
            "",
        )
        payload = json.loads(payload_line)
    except json.JSONDecodeError as exc:
        detail = stdout if stdout else "<no output>"
        return TTSRuntimeProbe(
            mlx_audio_version=fallback_version,
            probe_error=f"Invalid TTS runtime probe output ({exc}). Output: {detail}",
        )

    if not isinstance(payload, dict) or not isinstance(payload.get("available_model_types", []), list):
        return TTSRuntimeProbe(mlx_audio_version=fallback_version, probe_error="Invalid TTS runtime probe payload schema.")
    available = payload.get("available_model_types", [])
    remapping = payload.get("model_remapping", {})
    return TTSRuntimeProbe(
        mlx_audio_version=payload.get("mlx_audio_version") or fallback_version,
        available_model_types=tuple(
            sorted(str(item) for item in available if str(item).strip())
        ),
        model_remapping={
            str(key): str(value)
            for key, value in remapping.items()
            if str(key).strip() and str(value).strip()
        }
        if isinstance(remapping, dict)
        else {},
        probe_error=payload.get("probe_error"),
    )


def _supported_tts_model_types(probe: TTSRuntimeProbe) -> tuple[str, ...]:
    supported = {item for item in probe.available_model_types if item.strip()}
    supported.update(
        value.strip() for value in probe.model_remapping.values() if value.strip()
    )
    return tuple(sorted(supported))


def _tts_model_support_status(model_key: str, probe: TTSRuntimeProbe) -> str:
    if probe.probe_error:
        return "unknown"
    expected_model_type = canonical_model_type_for_key(model_key)
    if expected_model_type in _supported_tts_model_types(probe):
        return "supported"
    return "unsupported"


def validate_tts_model_runtime_support(
    model_key: str,
    *,
    expected_conda_env: str | None = None,
    timeout_seconds: int = 20,
    probe: TTSRuntimeProbe | None = None,
) -> TTSRuntimeProbe:
    resolved_expected_env = _resolve_expected_conda_env(expected_conda_env)
    probe = probe if probe is not None else probe_tts_runtime_support(timeout_seconds=timeout_seconds)
    installed_version = probe.mlx_audio_version or "unknown"
    expected_model_type = canonical_model_type_for_key(model_key)

    if probe.probe_error:
        raise RuntimeError(
            "Unable to verify mlx-audio TTS runtime compatibility for model "
            f"'{model_key}'.\n"
            f"Installed mlx-audio: {installed_version}\n"
            f"Probe error: {probe.probe_error}\n"
            "This usually indicates a local mlx-audio/MLX runtime issue before model loading.\n"
            f"{_required_command_hint(resolved_expected_env)}"
        )

    supported_model_types = _supported_tts_model_types(probe)
    if expected_model_type in supported_model_types:
        return probe

    raise RuntimeError(
        "Selected model "
        f"'{model_key}' is not supported by the active mlx-audio runtime.\n"
        f"Installed mlx-audio: {installed_version}\n"
        f"Expected upstream TTS model type: {expected_model_type}\n"
        "Detected upstream TTS model types: "
        f"{', '.join(supported_model_types) if supported_model_types else 'none'}\n"
        f"{_required_command_hint(resolved_expected_env)}"
    )


def validate_model_load_preflight(
    model_key: str,
    model_reference: str,
    *,
    expected_conda_env: str | None = None,
) -> None:
    resolved_expected_env = _resolve_expected_conda_env(expected_conda_env)

    if canonical_model_type_for_key(model_key) != "voxtral_tts":
        return

    model_path = Path(model_reference)
    if not model_path.exists() or not model_path.is_dir():
        return

    tekken_path = model_path / "tekken.json"
    if not tekken_path.exists():
        raise RuntimeError(
            "Resolved Voxtral model is missing its tokenizer file 'tekken.json'. "
            "The local snapshot may be incomplete; rerun the command to re-download the model.\n"
            f"{_required_command_hint(resolved_expected_env)}"
        )

    try:
        mistral_tokenizer_cls = _get_mistral_tokenizer_cls()
    except Exception as exc:
        mistral_common_version = _collect_package_versions(("mistral-common",)).get(
            "mistral-common"
        ) or "unknown"
        raise RuntimeError(
            "Voxtral requires the 'mistral-common' tokenizer package at runtime, but it could "
            f"not be imported.\nInstalled mistral-common: {mistral_common_version}\n"
            f"Import error: {type(exc).__name__}: {exc}\n"
            "Upgrade the active environment, then retry.\n"
            f"{_required_command_hint(resolved_expected_env)}"
        ) from exc

    try:
        mistral_tokenizer_cls.from_file(str(tekken_path))
    except TypeError as exc:
        mistral_common_version = _collect_package_versions(("mistral-common",)).get(
            "mistral-common"
        ) or "unknown"
        raise RuntimeError(
            "The active 'mistral-common' package cannot parse Voxtral's tokenizer config.\n"
            f"Installed mistral-common: {mistral_common_version}\n"
            f"Tokenizer file: {tekken_path}\n"
            f"Underlying error: {type(exc).__name__}: {exc}\n"
            "Upgrade 'mistral-common' in the active environment, then retry Voxtral.\n"
            f"{_required_command_hint(resolved_expected_env)}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to validate Voxtral tokenizer assets before model load.\n"
            f"Tokenizer file: {tekken_path}\n"
            f"Underlying error: {type(exc).__name__}: {exc}\n"
            f"{_required_command_hint(resolved_expected_env)}"
        ) from exc


def doctor(
    expected_conda_env: str | None = None,
    *,
    include_tts_runtime_probe: bool = False,
) -> EnvironmentReport:
    resolved_expected_env = _resolve_expected_conda_env(expected_conda_env)
    package_versions = _collect_package_versions(REQUIRED_PACKAGES)
    tts_runtime_probe = (
        probe_tts_runtime_support()
        if include_tts_runtime_probe
        else _empty_tts_runtime_probe(package_versions.get("mlx-audio"))
    )
    tts_model_support = (
        {
            model_key: _tts_model_support_status(model_key, tts_runtime_probe)
            for model_key in list_model_keys()
        }
        if include_tts_runtime_probe
        else {}
    )
    return EnvironmentReport(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        conda_env=os.getenv("CONDA_DEFAULT_ENV"),
        expected_conda_env=resolved_expected_env,
        package_versions=package_versions,
        tts_runtime_probe=tts_runtime_probe,
        tts_model_support=tts_model_support,
        python_prefix=sys.prefix,
        project_virtualenv=str(project_root() / ".venv"),
        is_project_virtualenv=is_project_virtualenv(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        ffmpeg_executable=shutil.which("ffmpeg"),
        is_local_ffmpeg=is_local_encoder(),
    )


def is_project_virtualenv() -> bool:
    return (
        Path(sys.prefix).absolute() == (project_root() / ".venv").absolute()
        and sys.prefix != sys.base_prefix
    )


def validate_project_virtualenv() -> None:
    if not is_project_virtualenv():
        raise RuntimeError(
            "Libro-TTS generation requires this project's .venv. "
            "Run bash scripts/setup.sh, then bash run.sh <args>.\n"
            f"Expected environment: {project_root() / '.venv'}\n"
            f"Current Python: {sys.executable} (prefix: {sys.prefix})"
        )


def is_local_encoder() -> bool:
    selected = shutil.which("ffmpeg")
    prefix = (project_root() / ".venv").resolve()
    return selected is not None and Path(selected).resolve().is_relative_to(prefix)


def validate_local_encoder() -> None:
    if not is_local_encoder():
        raise RuntimeError(
            "MP3/FLAC generation requires the project-local FFmpeg binary. "
            "Run bash scripts/setup.sh to install the locked encoder. "
            f"Selected encoder: {shutil.which('ffmpeg') or 'missing'}"
        )


def _required_command_hint(expected_conda_env: str | None) -> str:
    return (
        "Run commands via the Python environment where Libro-TTS dependencies are installed:\n"
        f"  {_runtime_command(expected_conda_env)}\n"
        f"Current Python: {sys.executable} ({platform.python_version()})"
    )


def validate_runtime_environment(
    expected_conda_env: str | None = None,
    skip_env_check: bool = False,
) -> EnvironmentReport:
    if not skip_env_check:
        validate_project_virtualenv()
    report = doctor(expected_conda_env=expected_conda_env, include_tts_runtime_probe=False)

    if (
        not skip_env_check
        and report.expected_conda_env is not None
        and report.conda_env != report.expected_conda_env
    ):
        raise RuntimeError(
            "This project is configured to run inside the conda environment "
            f"'{report.expected_conda_env}', but CONDA_DEFAULT_ENV is "
            f"'{report.conda_env}'.\n{_required_command_hint(expected_conda_env)}"
        )

    missing_imports: list[str] = []
    for module_name in ("mlx_audio", "huggingface_hub"):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_imports.append(module_name)

    if missing_imports:
        raise RuntimeError(
            "Missing required runtime modules: "
            + ", ".join(missing_imports)
            + "\n"
            + _required_command_hint(expected_conda_env)
        )

    unresolved_versions = [
        package for package, version in report.package_versions.items() if version is None
    ]
    if unresolved_versions:
        raise RuntimeError(
            "Missing required installed packages: "
            + ", ".join(unresolved_versions)
            + "\n"
            + _required_command_hint(expected_conda_env)
        )

    if not skip_env_check:
        prefix = Path(sys.prefix).resolve()
        for package in REQUIRED_PACKAGES:
            distribution = importlib.metadata.distribution(package)
            origin = Path(distribution.locate_file("")).resolve()
            if not origin.is_relative_to(prefix):
                raise RuntimeError(
                    f"Dependency '{package}' is outside the project environment: {origin}. "
                    "Use bash run.sh and rerun bash scripts/setup.sh."
                )

    return report


def validate_mlx_backend_preflight(
    *,
    expected_conda_env: str | None = None,
    skip_preflight: bool = False,
    timeout_seconds: int = 20,
) -> TTSRuntimeProbe | None:
    if skip_preflight:
        return None

    resolved_expected_env = _resolve_expected_conda_env(expected_conda_env)
    # The capability probe imports the same MLX TTS runtime in a child process.
    # Reuse that result for the subsequent selected-model compatibility check.
    probe = probe_tts_runtime_support(timeout_seconds=timeout_seconds)
    if not probe.probe_error:
        return probe
    raise RuntimeError(
        "MLX backend failed preflight before generation.\n"
        f"{probe.probe_error}\n"
        "Verify in the same shell:\n"
        f"  {_mlx_import_check_command(resolved_expected_env)}"
    )
