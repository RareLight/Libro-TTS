from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .catalog import canonical_model_type_for_key, list_model_keys

REQUIRED_PACKAGES = ("mlx-audio", "huggingface_hub")
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


def _collect_package_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def configure_local_cache_environment(hf_cache_root: Path) -> None:
    hf_home = hf_cache_root
    hub_cache = hf_home / "hub"
    transformers_cache = hf_home / "transformers"
    datasets_cache = hf_home / "datasets"

    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resolve_expected_conda_env(expected_conda_env: str | None = None) -> str | None:
    if expected_conda_env is not None:
        normalized = expected_conda_env.strip()
        return normalized or None

    configured = os.getenv("LIBRO_TTS_EXPECTED_CONDA_ENV", "").strip()
    return configured or None


def _runtime_command(expected_conda_env: str | None) -> str:
    if expected_conda_env:
        return f"conda run -n {expected_conda_env} python Libro-tts.py <args>"
    return "python Libro-tts.py <args>"


def _mlx_import_check_command(expected_conda_env: str | None) -> str:
    if expected_conda_env:
        return (
            f"conda run -n {expected_conda_env} python -c "
            "\"from mlx_audio.tts.utils import load_model; print('mlx-audio runtime import OK')\""
        )
    return 'python -c "from mlx_audio.tts.utils import load_model; print(\'mlx-audio runtime import OK\')"'


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
) -> TTSRuntimeProbe:
    resolved_expected_env = _resolve_expected_conda_env(expected_conda_env)
    probe = probe_tts_runtime_support(timeout_seconds=timeout_seconds)
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

    return report


def validate_mlx_backend_preflight(
    *,
    expected_conda_env: str | None = None,
    skip_preflight: bool = False,
    timeout_seconds: int = 20,
) -> None:
    if skip_preflight:
        return

    resolved_expected_env = _resolve_expected_conda_env(expected_conda_env)

    script = (
        "from mlx_audio.tts.utils import load_model\n"
        "print('mlx-audio runtime import OK')\n"
    )
    command = [sys.executable, "-c", script]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(os.environ),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"MLX backend preflight timed out after {timeout_seconds}s.\n"
            "Run commands via the Python environment where Libro-TTS dependencies are installed:\n"
            f"  {_runtime_command(resolved_expected_env)}\n"
            "Then verify:\n"
            f"  {_mlx_import_check_command(resolved_expected_env)}"
        ) from exc

    if result.returncode == 0:
        return

    details: list[str] = [
        "MLX backend failed preflight before generation.",
        f"Exit code: {result.returncode}",
    ]
    if result.stdout.strip():
        details.append("stdout:")
        details.append(_trim_process_output(result.stdout))
    if result.stderr.strip():
        details.append("stderr:")
        details.append(_trim_process_output(result.stderr))

    details.extend(
        [
            "",
            "This usually indicates a local mlx-audio/MLX runtime issue before model-specific compatibility checks.",
            "Verify in the same shell:",
            f"  {_mlx_import_check_command(resolved_expected_env)}",
        ]
    )
    raise RuntimeError("\n".join(details))
