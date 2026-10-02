#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${HOME}/.local/bin"
TARGET_PATH="${TARGET_DIR}/libro-tts"
SOURCE_PATH="${PROJECT_ROOT}/run.sh"

mkdir -p "$TARGET_DIR"
ln -sfn "$SOURCE_PATH" "$TARGET_PATH"

chmod +x "$SOURCE_PATH"

echo "Installed libro-tts -> $SOURCE_PATH"
echo "Command path: $TARGET_PATH"
