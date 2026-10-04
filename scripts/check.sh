#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
uv lock --check --offline
uv pip check --python .venv/bin/python
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -m ruff check libro_tts tests scripts Libro-tts.py
for script in run.sh scripts/*.sh; do
  bash -n "$script"
done
.venv/bin/python -B scripts/check_no_download.py
.venv/bin/python -B tests/integration_run_all_models.py
