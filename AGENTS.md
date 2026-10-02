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

Via conda wrapper: `bash run.sh input.txt -o output.wav`

### Testing

```bash
# Fast unit + mocked integration tests
python -m unittest discover -s tests -v

# Single test file
python -m unittest tests.test_cli -v

# Smoke all real models (requires downloaded models)
bash scripts/smoke_all_models.sh

# Full integration matrix (real generation)
python tests/integration_run_all_models.py --run
python tests/integration_run_all_models.py --run --offline-second-pass
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
| `runtime.py` | Text chunking, model loading, serial/parallel generation, audio concatenation |
| `catalog.py` | Model specs, voice presets, defaults (`DEFAULT_MODEL_KEY = "kokoro"`) |
| `store.py` | Model download, file-locked manifest at `models/resolved_models.json` |
| `env.py` | Conda/MLX environment validation, subprocess-isolated backend preflight |
| `paths.py` | Project root, models dir, HF cache paths |
| `text.py` | Text sanitization (footnotes, whitespace normalization) |
| `logging_config.py` | Logging setup, upstream warning suppression |

### Generation Pipeline (runtime.py)

1. Read input (UTF-8 → UTF-8-sig → Latin-1 fallback)
2. Sanitize text (`text.py`)
3. Sentence-aware chunking targeting 10–15 sec audio segments
4. Resolve model via `ModelStore.ensure_model()` (downloads if needed)
5. Serial OR parallel chunk generation (multiprocessing pool)
   - Parallel disabled for `csm` and `dia` models
   - Auto-falls back to serial on failure
6. Concatenate numpy audio arrays → write WAV/MP3/FLAC

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

Models are stored project-locally under `./models/` with HF cache redirected to `./models/.hf/`. This makes the full setup portable. The manifest at `./models/resolved_models.json` uses file locking to handle concurrent access.

### Environment Variables

- `LIBRO_TTS_EXPECTED_CONDA_ENV` — enforce a specific conda env name
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE` — auto-redirected to `./models/.hf/`
- `TOKENIZERS_PARALLELISM` — set to `"false"` internally

## Testing Patterns

- Framework: `unittest` (stdlib) with `unittest.mock`
- `test_mocked_pipeline.py` — tests the full generation pipeline without loading real models (heaviest test file)
- `integration_run_all_models.py` — real end-to-end runs, not part of the default `discover` suite
- MLX backend validation runs in a subprocess to avoid Metal initialization side effects
