#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-tts}"
INPUT_FILE="${INPUT_FILE:-$ROOT_DIR/tests_output/_smoke_input.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/tests_output/smoke_models}"
VOICE_NAME="${VOICE_NAME:-Emma}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

MODELS=(
  "kokoro"
  "qwen3_tts"
  "csm"
  "dia"
  "spark"
  "chatterbox"
  "soprano"
  "voxtral_tts"
)

PROMPT_MODELS=("csm" "dia" "spark" "chatterbox")

mkdir -p "$(dirname "$INPUT_FILE")" "$OUTPUT_DIR"

if [[ ! -f "$ROOT_DIR/reference_voices/${VOICE_NAME}.wav" ]]; then
  echo "Missing reference voice: $ROOT_DIR/reference_voices/${VOICE_NAME}.wav" >&2
  exit 1
fi

cat > "$INPUT_FILE" <<'TXT'
This is a short Libro-TTS smoke test. If you can hear this sentence clearly, the model run succeeded.
TXT

is_prompt_model() {
  local model="$1"
  for m in "${PROMPT_MODELS[@]}"; do
    [[ "$m" == "$model" ]] && return 0
  done
  return 1
}

build_cmd() {
  local model="$1"
  local out_file="$2"
  local -a cmd=(
    conda run -n "$CONDA_ENV"
    python Libro-tts.py "$INPUT_FILE"
    --model "$model"
    --output "$out_file"
  )
  if [[ "$SKIP_PREFLIGHT" == "1" ]]; then
    cmd+=(--skip-mlx-preflight)
  fi
  if is_prompt_model "$model"; then
    cmd+=(--voice "$VOICE_NAME")
  fi
  printf '%q ' "${cmd[@]}"
}

echo "Running smoke test for ${#MODELS[@]} models..."
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo

passed=0
failed=0

for model in "${MODELS[@]}"; do
  out_file="$OUTPUT_DIR/${model}.wav"
  log_file="$OUTPUT_DIR/${model}.log"
  echo "[$model] starting..."
  cmd="$(build_cmd "$model" "$out_file")"

  if eval "$cmd" >"$log_file" 2>&1; then
    if [[ -s "$out_file" ]]; then
      echo "[$model] PASS -> $out_file"
      ((passed+=1))
    else
      echo "[$model] FAIL (no audio output) -> see $log_file"
      ((failed+=1))
    fi
  else
    echo "[$model] FAIL -> see $log_file"
    ((failed+=1))
  fi
done

echo
echo "Smoke test complete. Passed: $passed, Failed: $failed"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
