#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Libro-TTS requires macOS on Apple Silicon." >&2
  exit 1
fi

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"
uv sync --project "$PROJECT_ROOT" --locked --python "${LIBRO_TTS_PYTHON:-3.12}" --link-mode copy "$@"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/setup_runtime_tools.py"
