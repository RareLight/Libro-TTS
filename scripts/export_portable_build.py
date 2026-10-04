#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import platform
import shutil
import subprocess
import sys


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
)
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
)


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
- Local `models/` data and Hugging Face cache files already stored with the project
- `reference_voices/`
- The project `pyproject.toml` and `uv.lock`, setup helpers, and launcher
- A `requirements.txt` exported from the lock for inspection

## Runtime expectations

- macOS on Apple Silicon
- Python 3.12 (exported using Python {platform.python_version()})
- uv for the locked installation; recreate `.venv` on the destination machine

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

- This export keeps model data project-local, so copying this folder preserves the local model cache.
- Offline synthesis requires all auxiliary tokenizer/codec/voice assets as well as primary weights.
- The remaining asset-portability migration is tracked in the source implementation checklist.
- Use `bash scripts/setup.sh` to install from the project lock; the export never freezes a shared environment.
"""


def _copy_path(relative_path: str, export_dir: Path) -> None:
    source = PROJECT_ROOT / relative_path
    target = export_dir / relative_path

    if source.is_dir():
        shutil.copytree(source, target, ignore=IGNORE_NAMES)
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
    # Resolve the locked dependency export before copying potentially large assets.
    requirements = _build_requirements_text()
    export_dir = args.output_dir.resolve()
    if export_dir.exists():
        raise SystemExit(f"Export directory already exists: {export_dir}")

    export_dir.mkdir(parents=True, exist_ok=False)

    for relative_path in COPY_PATHS:
        _copy_path(relative_path, export_dir)

    (export_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
    (export_dir / "README.md").write_text(_portable_readme_text(export_dir), encoding="utf-8")
    shutil.copy2(PROJECT_ROOT / "README.md", export_dir / "README.upstream.md")

    print(export_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
