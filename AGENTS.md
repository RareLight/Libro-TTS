# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Libro-TTS is a batch text-to-speech CLI for Apple Silicon Macs. It uses MLX-backed models to convert `.txt` files to audio (WAV/MP3/FLAC), with sentence-aware chunking, optional parallel processing, and project-local model caching for portability.

## Commands

### Running

```bash
# Single file (default: Kokoro model)
python Libro-tts.py input.txt -o output.wav

# With specific model
python Libro-tts.py input.txt -o output --model qwen3_tts

# Batch directory
python Libro-tts.py --input-dir chapters -o output_dir

# Offline (no downloads)
python Libro-tts.py input.txt -o output --offline

# Diagnostics / list models
python Libro-tts.py --diag
python Libro-tts.py --list-models
python Libro-tts.py --list-kokoro-voices
```

Set up with `bash scripts/setup.sh`, then run with `bash run.sh input.txt -o output.wav`.
Python commands below must use `.venv/bin/python` (or an activated project `.venv`). The launcher always selects the project environment.

### Testing

```bash
# Fast unit + mocked integration tests
.venv/bin/python -m unittest discover -s tests -v

# Complete fast maintenance gate after development setup
bash scripts/check.sh

# Single test file
.venv/bin/python -m unittest tests.test_cli -v

# Smoke all real models (requires downloaded models)
bash scripts/smoke_all_models.sh

# Full integration matrix (real generation)
.venv/bin/python tests/integration_run_all_models.py --run
.venv/bin/python tests/integration_run_all_models.py --run --offline-second-pass
```

## Architecture

### Entry Points

- **`Libro-tts.py`** — top-level script; calls `libro_tts.cli.main()`
- **`libro_tts/cli.py`** — parses args, runs validation stack, dispatches to `process_single_file()` or `process_batch_dir()`
- **`libro_tts/runtime.py`** — all generation logic (text chunking → model loading → audio output)

### Module Responsibilities

| Module | Role |
|---|---|
| `cli.py` | Argument parsing, validation orchestration |
| `validation.py` | Backend-free request checks, canonical destination keys, output suffix policy |
| `runtime.py` | Text chunking, model loading, serial/parallel generation, audio concatenation |
| `audio_output.py` | Atomic sibling transactions and ordered incremental PCM16 WAV output |
| `diagnostics.py` | Bounded diagnostic tail capture for suppressed model output |
| `catalog.py` | Model specs, voice presets, defaults (`DEFAULT_MODEL_KEY = "kokoro"`) |
| `store.py` | Resumable staged model acquisition, primary asset validation, atomic shared manifest transactions |
| `assets.py` | Pinned auxiliary acquisition, project HF cache reuse, local-only runtime loading |
| `env.py` | Project virtualenv validation, subprocess-isolated backend preflight and diagnostics |
| `bootstrap.py` | Standard-library-only cache/offline policy before runtime imports |
| `paths.py` | Project root, models dir, HF cache paths |
| `text.py` | Text sanitization (footnotes, whitespace normalization) |
| `logging_config.py` | Logging setup, upstream warning suppression |

### Generation Pipeline (runtime.py)

1. Validate request/destinations and read input (CLI: strict UTF-8 with optional BOM; runtime API: configurable encodings with legacy Latin-1 fallback)
2. Sanitize text (`text.py`)
3. Sentence-aware chunking targeting 10–15 sec audio segments
4. Resolve model via `ModelStore.ensure_model()` (downloads if needed)
5. Serial OR parallel chunk generation (multiprocessing pool)
   - Parallel disabled for `csm` and `dia` models
   - Select execution policy before loading; parent model loads only for serial work/fallback
   - Auto-falls back to serial on failure
   - Discards partial parallel audio and resets chunk progress before retrying the full input serially
6. WAV: write ordered chunks incrementally as PCM16 to a sibling temporary file. Other formats: concatenate arrays and use the pinned encoder. Synchronize and atomically publish only complete output.

Execution controls: `--serial` or `--workers N`, with `--parallel-batch-size N`. Defaults remain two workers, a three-chunk threshold, and 12 chunks per pool batch. Pools remain per input. A batch-scoped reference context reuses successful transcript preparation, invalidating on model/store/audio/sibling changes. One isolated probe serves both backend and selected-family preflight.

Batch processing plans every output before generation and rejects colliding destinations, including case/Unicode variants and existing symlink aliases. Successful output overwrites remain intentional; failures preserve prior complete audio. Output safety regression coverage is in `tests/test_output_safety.py`.

Empty text after sanitization is rejected before model acquisition. Reference transcript resolution runs once per generation, records explicit/file/STT provenance, and shares normalized text across chunks. Missing/empty siblings trigger managed STT; failed Spark long-file recovery retains trimmed original text. Request and transcript edge cases are covered in `tests/test_request_validation.py` and `tests/test_reference_transcripts.py` without real models.

Audio segments require finite real samples, positive integral sample rates, and consistent channel layouts within and between text chunks. Legacy row/column mono shapes are retained; `(1, 1)` remains a vector, and float16 is promoted for the pinned encoder. Progress advances only after a chunk is accepted. `tests/test_audio_consistency.py` covers these boundaries and real spawned pool ordering/failure/fallback using pure fixtures without TTS models or Metal.

### Key Constants (runtime.py)

```python
DEFAULT_PARALLEL_WORKERS = 2
MIN_PARALLEL_CHUNK_COUNT = 3       # Only parallelize if ≥3 chunks
PARALLEL_DISABLED_MODEL_KEYS = {"csm", "dia"}
TARGET_CHUNK_DURATION_SECONDS = 12.5
BASE_CHARS_PER_SECOND = 12.0
```

### Supported Models

`kokoro` (default), `qwen3_tts`, `csm`, `dia`, `spark`, `chatterbox`, `soprano`, `voxtral_tts`

### Model Storage

Models are stored project-locally under `./models/` with HF cache redirected to `./models/.hf/`. New acquisitions use resumable staging at `models/.staging/` and publish only after validation. Manifest version 2 stores relative paths at `models/resolved_models.json`; a shared transaction lock and atomic JSON replacement protect concurrent updates. Primary entries (`models`) and auxiliary entries (`assets`) have separate completion scopes and `.libro-model.json` inventories. Acquire auxiliaries before entering `local_asset_loading`; it blocks HF network access during runtime loading and synthesis. Worker tasks carry the active cache root. See `docs/model-storage.md` for asset coverage and migration rules.

### Environment Variables

- `LIBRO_TTS_EXPECTED_CONDA_ENV` — enforce a specific conda env name
- `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_XET_CACHE`, `HF_ASSETS_CACHE`, `TRANSFORMERS_CACHE` — auto-redirected under `./models/.hf/` before runtime imports
- `TOKENIZERS_PARALLELISM` — set to `"false"` internally

## Testing Patterns

- Framework: `unittest` (stdlib) with `unittest.mock`
- `test_mocked_pipeline.py` — tests the full generation pipeline without loading real models (heaviest test file)
- `integration_run_all_models.py` — real end-to-end runs, not part of the default `discover` suite
- MLX backend validation runs in a subprocess to avoid Metal initialization side effects

Dependencies are specified in `pyproject.toml` and resolved in `uv.lock`. Setup installs into `.venv`; generation never installs packages. FFmpeg is provided locally by the locked `imageio-ffmpeg` wheel. Track remaining correctness/asset/performance work in `docs/implementation-checklist.md`.

Catalog fields declare prompt/reference/parallel policy and pinned speed behavior. Narrow `MODEL_CONFIG_TYPE_ADAPTERS` handle known snapshot backbone labels; reject unknown architectures instead of relabeling them. `RuntimeOptions.model_path_override` is internal only: local directories use explicit overrides, repo IDs pass through managed acquisition. `stream_wav=False` retains the buffered comparison path internally.

Maintenance/CI commands and update gates are in `docs/maintenance.md`; measured resource decisions are in `docs/performance-2026-10-03.md`. Fast tests use disposable reference fixtures and require no private `reference_voices/`. Heavy all-model integration remains opt-in; Whisper/Chatterbox verification and release relocation are outside the current scope.
