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

export LIBRO_TTS_EXPECTED_CONDA_ENV=tts

if [ "$#" -eq 0 ]; then
  set -- --help
fi

exec /Users/anna/miniconda3/bin/conda run --no-capture-output -n tts \
  python "$PROJECT_ROOT/Libro-tts.py" "$@"
