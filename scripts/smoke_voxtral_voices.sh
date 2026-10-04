#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_FILE="${INPUT_FILE:-$ROOT_DIR/tests_output/_voxtral_voice_input.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/tests_output/voxtral_voice_sweep}"
OFFLINE="${OFFLINE:-0}"

VOICES=(
  "casual_male"
  "casual_female"
  "cheerful_female"
  "neutral_male"
  "neutral_female"
)

mkdir -p "$(dirname "$INPUT_FILE")" "$OUTPUT_DIR"

cat > "$INPUT_FILE" <<'TXT'
This is a short Voxtral voice sweep for Libro-TTS. Each rendered sample should sound clear, stable, and noticeably distinct from the others.
TXT

echo "Running Voxtral voice sweep for ${#VOICES[@]} voices..."
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo

passed=0
failed=0

for voice in "${VOICES[@]}"; do
  out_file="$OUTPUT_DIR/${voice}.wav"
  log_file="$OUTPUT_DIR/${voice}.log"
  cmd=(
    bash "$ROOT_DIR/run.sh" "$INPUT_FILE"
    --model voxtral_tts
    --voice "$voice"
    --output "$out_file"
  )
  if [[ "$OFFLINE" == "1" ]]; then
    cmd+=(--offline)
  fi

  echo "[${voice}] starting..."
  if "${cmd[@]}" >"$log_file" 2>&1; then
    if [[ -s "$out_file" ]]; then
      echo "[${voice}] PASS -> $out_file"
      ((passed+=1))
    else
      echo "[${voice}] FAIL (no audio output) -> see $log_file"
      ((failed+=1))
    fi
  else
    echo "[${voice}] FAIL -> see $log_file"
    ((failed+=1))
  fi
done

echo
echo "Voxtral voice sweep complete. Passed: $passed, Failed: $failed"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
