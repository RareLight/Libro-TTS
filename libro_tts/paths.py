from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return project_root() / "models"


def hf_cache_dir() -> Path:
    return models_dir() / ".hf"


def resolved_manifest_path() -> Path:
    return models_dir() / "resolved_models.json"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_runtime_dirs() -> dict[str, Path]:
    model_root = ensure_dir(models_dir())
    cache_root = ensure_dir(hf_cache_dir())
    return {
        "models_dir": model_root,
        "hf_cache_dir": cache_root,
        "manifest_path": resolved_manifest_path(),
    }
