#!/usr/bin/env bash
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  TARGET="$(readlink "$SOURCE")"
  if [[ "$TARGET" == /* ]]; then
    SOURCE="$TARGET"
  else
    SOURCE="$SCRIPT_DIR/$TARGET"
  fi
done

PROJECT_ROOT="$(cd -P "$(dirname "$SOURCE")" && pwd)"

if [ "$#" -eq 0 ]; then
  set -- --help
fi

PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Libro-TTS's local environment is missing. Run: bash \"$PROJECT_ROOT/scripts/setup.sh\"" >&2
  exit 1
fi

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
exec "$PYTHON" "$PROJECT_ROOT/Libro-tts.py" "$@"
