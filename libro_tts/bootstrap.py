"""Process configuration that must run before importing runtime dependencies."""

from __future__ import annotations

import os
from pathlib import Path

from .paths import hf_cache_dir, project_root


def configure_local_cache_environment(
    hf_cache_root: Path | None = None,
    *,
    offline: bool = False,
    create_dirs: bool = False,
) -> None:
    root = (hf_cache_root if hf_cache_root is not None else hf_cache_dir()).resolve()
    cache_paths = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "hub",
        "HF_XET_CACHE": root / "xet",
        "HF_ASSETS_CACHE": root / "assets",
        "TRANSFORMERS_CACHE": root / "transformers",
        "HF_DATASETS_CACHE": root / "datasets",
    }
    for name, path in cache_paths.items():
        os.environ[name] = str(path)
        if create_dirs:
            path.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONNOUSERSITE"] = "1"
    local_bin = str(project_root() / ".venv" / "bin")
    entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry != local_bin]
    os.environ["PATH"] = os.pathsep.join([local_bin, *entries])
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
