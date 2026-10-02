#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
from pathlib import Path
import platform
import re
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
)
IGNORE_NAMES = shutil.ignore_patterns(
    ".DS_Store",
    "__pycache__",
    "*.pyc",
    "*.pyo",
)
REQUIREMENT_NAME_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)")


def _build_requirements_text() -> str:
    lines = [
        "# Generated from the active Libro-TTS runtime environment",
        f"# Python {platform.python_version()} on {platform.platform()}",
        "",
    ]

    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    requirement_lines: dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        normalized_line = line
        if " @ file://" in line:
            package_name = line.split(" @ ", 1)[0].strip()
            normalized_line = f"{package_name}=={importlib.metadata.version(package_name)}"

        match = REQUIREMENT_NAME_PATTERN.match(normalized_line)
        if not match:
            continue

        requirement_lines[match.group(1).lower()] = normalized_line

    lines.extend(requirement_lines.values())

    lines.append("")
    return "\n".join(lines)


def _portable_readme_text(export_dir: Path) -> str:
    return f"""# Libro-TTS Portable Build

This folder is a copyable Libro-TTS release bundle exported from the source repo on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

## What is included

- The runnable app entrypoint: `Libro-tts.py`
- The `libro_tts/` package
- Local `models/` data and Hugging Face cache files already stored with the project
- `reference_voices/`
- A fully pinned `requirements.txt` generated from the runtime environment used to export this build

## Runtime expectations

- macOS on Apple Silicon
- Python {platform.python_version()} is the recommended baseline because the requirements were captured from that runtime
- The destination machine should create its own venv or conda env, then install `requirements.txt`

## Setup

### venv

```bash
cd /path/to/{export_dir.name}
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### conda

```bash
cd /path/to/{export_dir.name}
conda create -n libro-tts python={platform.python_version()}
conda activate libro-tts
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python Libro-tts.py --diag
python Libro-tts.py your_text_file.txt -o output/book_ch1
```

The app no longer requires a conda env to be named `tts`; it runs in whatever Python environment has these dependencies installed.
At startup, Libro-TTS validates that the active `mlx-audio` runtime can actually support the selected TTS model family before attempting any download or model load.

## Notes

- This export keeps model data project-local, so copying this folder preserves the local model cache.
- If you want a fully offline run on the destination machine, keep the `models/` folder intact.
- `requirements.txt` is an exact environment capture, so installing into a clean Python {platform.python_version()} environment is the safest path.
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

    export_dir = args.output_dir.resolve()
    if export_dir.exists():
        raise SystemExit(f"Export directory already exists: {export_dir}")

    export_dir.mkdir(parents=True, exist_ok=False)

    for relative_path in COPY_PATHS:
        _copy_path(relative_path, export_dir)

    (export_dir / "requirements.txt").write_text(_build_requirements_text(), encoding="utf-8")
    (export_dir / "README.md").write_text(_portable_readme_text(export_dir), encoding="utf-8")
    shutil.copy2(PROJECT_ROOT / "README.md", export_dir / "README.upstream.md")

    print(export_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
