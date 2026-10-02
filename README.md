# Libro-TTS

Batch-oriented text-to-speech generation on Apple Silicon using [`mlx-audio`](https://github.com/Blaizzy/mlx-audio), with project-local model storage for portable distribution.

## Environment Contract

Run Libro-TTS from any Python environment that has the required dependencies installed.

- Supported runtime command patterns:
  - `python Libro-tts.py ...`
  - `conda run -n <your-env> python Libro-tts.py ...`
- If you want to enforce a specific conda env name locally, set `LIBRO_TTS_EXPECTED_CONDA_ENV=<your-env>`.
- Before generation, the CLI runs an MLX backend preflight in a subprocess so Metal/MLX initialization errors fail cleanly instead of aborting the main process.
- Before any model is downloaded or loaded, Libro-TTS also verifies that the active `mlx-audio` runtime supports the selected TTS model family.
- Verified local setup as of April 1, 2026: `conda run -n tts ...` with `mlx-audio 0.4.3`.
- If you want to use the verified local wrapper directly, run `./run.sh ...` (or the installed `libro-tts` symlink from `install_command.sh`).

## Requirements

- macOS on Apple Silicon.
A Python environment with:
  - packages from `requirements.txt` or at minimum `mlx-audio`, `mlx`, and `huggingface_hub`

Use diagnostics:

```bash
python Libro-tts.py --diag
```

## Supported TTS Models

The app supports these `mlx-audio` model families:

- `kokoro`
- `qwen3_tts`
- `csm`
- `dia`
- `spark`
- `chatterbox`
- `soprano`
- `voxtral_tts`

List current pinned candidates and defaults:

```bash
python Libro-tts.py --list-models
```

List Kokoro English voices:

```bash
python Libro-tts.py --list-kokoro-voices
```

Voxtral English voices supported by Libro-TTS:

- `casual_male`
- `casual_female`
- `cheerful_female`
- `neutral_male`
- `neutral_female`

Model preset lookup is restricted to MLX variants (`mlx-community/*`) so non-MLX repos are not selected by fallback.

At runtime, Libro-TTS uses the documented Python API pattern:

- `from mlx_audio.tts.utils import load_model`
- `model = load_model(<local_or_repo_model_path>)`
- `for result in model.generate(...): ...`

Generated chunk audio (`result.audio`) is concatenated in order and written as one contiguous output file per input text file.

## Local Model Storage and First-Run Behavior

On first use of a model preset, Libro-TTS downloads model assets into:

- `./models/<model_key>/<repo_id_sanitized>/`

A manifest is written at:

- `./models/resolved_models.json`

Subsequent runs reuse local model files instead of downloading again.

## Local Cache Portability

To keep assets portable and project-scoped, runtime cache roots are redirected under:

- `./models/.hf/`

This includes Hugging Face/transformers cache paths used during model and auxiliary asset resolution.

## CLI Usage

### Single file (default model/settings)

```bash
python Libro-tts.py input.txt -o tests_output/book_ch1
```

Default output is user-focused and concise (model/voice/speed/lang and save path), with an animated chunk progress bar during synthesis.
Use `--verbose` to enable full diagnostic logging.
Long text is automatically split into sentence-aware synthesis chunks targeting roughly 10-15 seconds of audio each, then reassembled into one contiguous output file.
If `--output` includes a file extension (for example `chapter.wav`), Libro-TTS uses that path directly.
If no extension is provided, Libro-TTS appends `.<audio-format>`.

### Single file with model override

```bash
python Libro-tts.py input.txt -o tests_output/qwen --model qwen3_tts
```

### Single file with Voxtral TTS

```bash
python Libro-tts.py input.txt -o tests_output/voxtral --model voxtral_tts --voice neutral_female
```

### Batch mode

```bash
python Libro-tts.py --input-dir chapters -o tests_output/chapters --model kokoro
```

Batch mode treats `.txt` extension matching case-insensitively (for example `chapter.TXT`).
Batch exits non-zero when any file fails, so CI/scripting can reliably detect partial failures.

### Offline mode (no network)

```bash
python Libro-tts.py input.txt -o tests_output/offline --model kokoro --offline
```

### Diagnostics

```bash
python Libro-tts.py --diag
```

## Behavioral Defaults

- Default model key: `kokoro`
- Default Kokoro voice: `af_aoede`
- Default Kokoro speed: `1.1`
- Default Qwen3 voice: `Vivian`
- Default Voxtral voice: `neutral_female`
- English-only first release language mapping:
  - Kokoro: `a` (American English code used by Kokoro)
  - Qwen3-TTS: `english`
  - Voxtral TTS: no `lang_code` override is sent; the selected named voice controls the English preset
  - CSM / Dia / Spark / Chatterbox / Soprano: `en`
- Input text is expected to be UTF-8 encoded.
- Prompt-based model defaults (`csm`, `dia`, `spark`, `chatterbox`): `./reference_voices/Britney.wav`
- Named-voice models (`kokoro`, `qwen3_tts`, `voxtral_tts`) use upstream voice names via `--voice`, not `./reference_voices`.
- If you do not pass alternate `--model/--voice/--speed`, defaults are preserved.
- Prompt-based models accept `--voice` as either:
  - a direct `.wav` file path, or
  - a prompt name from `./reference_voices` (for example `--voice Britney`).
- For cloning models that use transcripts (`csm`, `dia`, `spark`), Libro-TTS also looks for a matching
  `.txt` file beside the selected `.wav` (for example `Britney.wav` + `Britney.txt`).

## Testing

### Model compatibility and diagnostics

```bash
python Libro-tts.py --diag
```

`--diag` reports package versions, the probed upstream `mlx-audio` TTS model families, and Libro-TTS compatibility status for each configured model key.

### All-model smoke run

```bash
bash scripts/smoke_all_models.sh
```

### Voxtral English voice sweep

```bash
bash scripts/smoke_voxtral_voices.sh
```

Optional offline Voxtral reuse check after the first successful download:

```bash
OFFLINE=1 bash scripts/smoke_voxtral_voices.sh
```

### Fast unit and mocked integration checks

```bash
python -m unittest discover -s tests -v
```

### Heavy real integration matrix (opt-in)

Uses `input.txt` and public-domain fixture text files to run generation across all supported models.

```bash
python tests/integration_run_all_models.py --run
```

Optional second pass to verify offline local reuse:

```bash
python tests/integration_run_all_models.py --run --offline-second-pass
```

## Notes

- Quality-first policy: model conversion/downloading is full precision only (no quantization).
- Batch generation favors output quality and consistency over low-latency streaming.
