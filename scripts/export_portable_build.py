#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import replace
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_ROOT = PROJECT_ROOT / "dist"
COPY_PATHS = (
    "Libro-tts.py",
    "libro_tts",
    "models",
    "reference_voices",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "run.sh",
    "scripts/setup.sh",
    "scripts/setup_runtime_tools.py",
    "LICENSE",
    "docs",
)
OPTIONAL_PATHS = {"models", "reference_voices"}
IGNORE_NAMES = shutil.ignore_patterns(
    ".DS_Store",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "token",
    "stored_tokens",
    "*.lock",
    ".locks",
    "*.incomplete",
    ".staging",
    "*.log",
    "*.tmp",
)


def _ignored_names(directory: str, names: list[str]) -> set[str]:
    ignored = set(IGNORE_NAMES(directory, names))
    if Path(directory).name == ".hf":
        ignored.update({"xet"} & set(names))
    return ignored


def _source_links(source: Path) -> list[tuple[Path, Path]]:
    """Validate copied links before copying; never traverse an external tree."""
    root = source.resolve(strict=True)
    if not root.is_relative_to(PROJECT_ROOT.resolve()):
        raise RuntimeError(f"Export source escapes the project: {source}")
    links = []
    if source.is_dir():
        for directory, directories, files in os.walk(source, followlinks=False):
            ignored = _ignored_names(directory, directories + files)
            directories[:] = [name for name in directories if name not in ignored]
            for name in directories + [name for name in files if name not in ignored]:
                path = Path(directory) / name
                if not path.is_symlink():
                    continue
                try:
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise RuntimeError(f"Broken or cyclic export symlink: {path}") from exc
                if not resolved.is_relative_to(root):
                    raise RuntimeError(f"Export symlink escapes its source tree: {path}")
                relative = resolved.relative_to(root)
                if ".cache" in relative.parts or (".hf", "xet") in tuple(zip(relative.parts, relative.parts[1:])) or _ignored_names(str(resolved.parent), list(relative.parts)):
                    raise RuntimeError(f"Export symlink targets excluded metadata: {path}")
                links.append((path.relative_to(source), relative))
    return links


def _normalize_model_manifest(export_dir: Path) -> list[str]:
    """Adopt only included models offline, preserving known local provenance."""
    root = export_dir / "models"
    if not root.is_dir():
        return []
    # Standalone exporter invocation starts with scripts/ on sys.path.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from libro_tts.catalog import list_model_specs
    from libro_tts.store import ModelStore, _atomic_write_json

    store = ModelStore(root)
    original = store._load_manifest()
    included = []
    for spec in list_model_specs():
        entry = original["models"].get(spec.key)
        if entry and entry["repo_id"] not in spec.repo_candidates:
            spec = replace(spec, repo_candidates=(entry["repo_id"], *spec.repo_candidates))
            store._validate_mlx_repo_candidates(spec)
        # Existing-snapshot resolution performs no acquisition or model loading.
        if store._resolve_existing(spec) is not None:
            included.append(spec.key)
    manifest = store._load_manifest()
    manifest["version"] = 2
    manifest["models"] = {key: entry for key, entry in manifest["models"].items() if key in included}
    assets = {}
    for key, entry in manifest.get("assets", {}).items():
        path = Path(entry["path"])
        if path.is_absolute():
            try:
                path = path.relative_to(PROJECT_ROOT / "models")
            except ValueError:
                revision = entry.get("revision")
                if not revision:
                    continue
                path = Path(".hf/hub") / ("models--" + entry["repo_id"].replace("/", "--")) / "snapshots" / revision
        candidate = store._contained(root / path)
        if candidate.is_dir():
            assets[key] = {**entry, "path": candidate.relative_to(root).as_posix()}
    if "assets" in manifest:
        manifest["assets"] = assets
    _atomic_write_json(store.manifest_path, manifest)
    return included


def _prune_metadata(export_dir: Path) -> None:
    # Keep download metadata until primary snapshot revisions have been captured.
    for directory, directories, files in os.walk(export_dir / "models", followlinks=False):
        ignored = _ignored_names(directory, directories + files) | ({".cache"} & set(directories))
        for name in ignored:
            path = Path(directory) / name
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        directories[:] = [name for name in directories if name not in ignored]


def _build_requirements_text() -> str:
    result = subprocess.run(
        ["uv", "--no-cache", "export", "--project", str(PROJECT_ROOT), "--frozen",
         "--no-dev", "--no-emit-project"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _portable_readme_text(export_dir: Path) -> str:
    return f"""# Libro-TTS Portable Build

This folder is a copyable Libro-TTS release bundle exported from the source repo on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

## What is included

- The runnable app entrypoint: `Libro-tts.py`
- The `libro_tts/` package
- Available local `models/` snapshots and required Hugging Face cache data
- `reference_voices/`, when present (including your private reference audio/text)
- The project `pyproject.toml` and `uv.lock`, setup helpers, and launcher
- A `requirements.txt` exported from the lock for inspection
- Project license, model cards/notices already present, and source-reference documentation

## Runtime expectations

- macOS on Apple Silicon
- Python 3.12 (exported using Python {platform.python_version()})
- uv for the locked installation; recreate `.venv` on the destination machine
- Internet access for initial locked dependency/Python installation; cached model generation can then run offline

## Setup

```bash
cd /path/to/{export_dir.name}
bash scripts/setup.sh --no-dev
```

## Run

```bash
bash run.sh --diag
bash run.sh your_text_file.txt -o output/book_ch1
```

The launcher runs this project's `.venv`; generation rejects other environments.
At startup, Libro-TTS validates that the active `mlx-audio` runtime can actually support the selected TTS model family before attempting any download or model load.

## Notes

- Move/copy the bundle before setup. If moving it after setup, recreate `.venv` at the new location rather than copying the old environment.
- Exported manifest paths are relative; absent historical model entries are removed. Internal cache symlinks are made relative; external or broken links fail export.
- Credentials, partial downloads, locks, download metadata, and Xet logs/chunks are excluded. Primary revision inventories are captured before discarding download metadata.
- Offline synthesis requires all auxiliary tokenizer/codec/voice assets as well as primary weights.
- Only included validated primary snapshots are recorded in the manifest. Generation cannot use an absent model offline.
- The source-reference documentation describes development/release gates; the minimal runtime bundle does not include the source checkout's test suite.
- Application/model notices are retained; a dependency or model redistribution review is a separate gate before public distribution.
- Use `bash scripts/setup.sh` to install from the project lock; the export never freezes a shared environment.
"""


def _copy_path(relative_path: str, export_dir: Path) -> None:
    source = PROJECT_ROOT / relative_path
    target = export_dir / relative_path
    if relative_path in OPTIONAL_PATHS and not source.exists() and not source.is_symlink():
        return

    links = _source_links(source)

    if source.is_dir():
        shutil.copytree(source, target, ignore=_ignored_names, symlinks=True)
        for relative_link, relative_target in links:
            link = target / relative_link
            link.unlink()
            link.symlink_to(os.path.relpath(target / relative_target, link.parent))
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _default_export_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return EXPORT_ROOT / f"Libro-TTS-portable-{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a portable Libro-TTS release folder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_export_dir(),
        help="Destination directory for the exported portable build.",
    )
    args = parser.parse_args()

    if Path(sys.prefix).absolute() != PROJECT_ROOT / ".venv" or sys.prefix == sys.base_prefix:
        raise SystemExit("Run the exporter using this project's .venv/bin/python.")
    export_dir = args.output_dir.resolve()
    if export_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Export directory already exists: {export_dir}")
    # Prevent recursive copying when output is placed under an included tree.
    for relative_path in COPY_PATHS:
        source = PROJECT_ROOT / relative_path
        if source.is_dir() and export_dir.is_relative_to(source.resolve()):
            raise SystemExit(f"Export destination is inside an included source tree: {source}")
        if relative_path not in OPTIONAL_PATHS or source.exists() or source.is_symlink():
            _source_links(source)
    requirements = _build_requirements_text()
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{export_dir.name}.tmp-", dir=export_dir.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for relative_path in COPY_PATHS:
            _copy_path(relative_path, staged)
        _normalize_model_manifest(staged)
        _prune_metadata(staged)
        (staged / "requirements.txt").write_text(requirements, encoding="utf-8")
        (staged / "README.md").write_text(_portable_readme_text(export_dir), encoding="utf-8")
        shutil.copy2(PROJECT_ROOT / "README.md", staged / "README.upstream.md")
        if export_dir.exists() or export_dir.is_symlink():
            raise SystemExit(f"Export directory already exists: {export_dir}")
        staged.rename(export_dir)

    print(export_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
