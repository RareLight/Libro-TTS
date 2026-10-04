# Libro-TTS

Batch-oriented text-to-speech generation on Apple Silicon using [`mlx-audio`](https://github.com/Blaizzy/mlx-audio), with project-local model storage for portable distribution.

## Environment Contract

Run Libro-TTS through its project-local virtual environment. Install the locked dependencies explicitly before generation:

```bash
bash scripts/setup.sh
bash run.sh --diag
```

Setup requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12. It creates `.venv` and installs `uv.lock`; use `bash scripts/setup.sh --no-dev` for a runtime-only installation. An existing Python 3.12 interpreter can be selected with `LIBRO_TTS_PYTHON=/path/to/python3.12`. Setup and synthesis are separate: generation never installs or upgrades Python packages.

- Supported runtime command patterns:
  - `bash run.sh ...`
  - `.venv/bin/python Libro-tts.py ...`
- Generation checks the interpreter's project environment identity. Activating an unrelated Conda/venv environment does not change the launcher's interpreter.
- Before generation, one isolated subprocess checks the MLX backend and available TTS families, so Metal/import errors fail cleanly before acquisition. The selected-model check reuses that probe.
- The lock preserves the inspected `mlx-audio 0.4.3` Git revision `e42e1431fcf89af313375296c46d03a0153c4aa7`, rather than substituting PyPI's different distribution with that version number.
- `run.sh` also supports the installed `libro-tts` symlink from `install_command.sh` and preserves the caller's working directory.

## Requirements

- macOS on Apple Silicon.
- Python 3.12 and the packages resolved in `uv.lock`, including TTS/STT extras, the English spaCy pipeline, and Voxtral tokenizer support.
- FFmpeg supplied by the pinned `imageio-ffmpeg` wheel inside `.venv`, exposed as `.venv/bin/ffmpeg` by setup. MP3/FLAC generation rejects external encoder fallback.
- macOS Metal and system frameworks remain platform prerequisites.

The encoder wheel is provided by [imageio-ffmpeg 0.6.0](https://pypi.org/project/imageio-ffmpeg/0.6.0/). The verified arm64 binary is FFmpeg 7.1; `.venv/bin/ffmpeg -L` prints its GPL notice and build configuration. Runtime-bundle redistribution and license/source delivery remain release acceptance gates in the checklist.

Use diagnostics:

```bash
bash run.sh --diag
```

The remaining examples using `python` assume `source .venv/bin/activate`; using `bash run.sh` avoids activation entirely.

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
- `model = load_model(<local_model_path>)`
- `for result in model.generate(...): ...`

Generated chunks (`result.audio`) retain text order in one output file per input. WAV writes PCM16 incrementally into an atomic staging file; other formats retain the buffered upstream encoder.

## Local Model Storage and First-Run Behavior

On first use of a model preset, Libro-TTS downloads model assets into:

- `./models/<model_key>/<repo_id_sanitized>/`

A manifest is written at:

- `./models/resolved_models.json`

Subsequent runs reuse local model files instead of downloading again.
New downloads use resumable staging under `models/.staging/`, pin the resolved repository commit, and validate the complete file inventory plus required primary tokenizer/voice/codec assets before publishing. Manifest and config updates use atomic replacement; a shared manifest lock protects updates across different model keys.

Manifest version 2 stores paths relative to the active model root. Legacy entries are adopted from validated candidates in the current project, preserving weights and recording known revisions without following another checkout's absolute paths. Each validated snapshot has a `.libro-model.json` completion inventory. Primary models and separate auxiliary repositories have distinct completion scopes. See [model storage and asset coverage](docs/model-storage.md) for recovery behavior and validation limits.

## Local Cache Portability

To keep assets portable and project-scoped, runtime cache roots are redirected under:

- `./models/.hf/`

This includes Hugging Face/transformers cache paths used during model and auxiliary asset resolution.
Cache configuration runs before runtime imports and includes the current hub, Xet, and asset cache variables. `--offline` sets HF/transformers offline policy before imports and preflight. Keep models separate from `.venv`: rebuilding the environment does not remove downloaded model assets.

CSM's text tokenizer/Mimi codec, Dia's DAC codec, and Chatterbox's S3Tokenizer are acquired explicitly before loading their TTS runtimes. Whisper is acquired only when a reference transcript needs transcription. These repositories live in `models/.hf/hub/`, use pinned resumable acquisition, and have separate `assets` manifest entries. Tokenizer-only acquisition excludes Llama language-model weights. Existing project-cache snapshots are reused after validation. Runtime loading and synthesis use local-only HF policy, even after online acquisition, so upstream loaders cannot silently fetch newer or untracked assets. A missing offline dependency produces a local-asset error.

Kokoro named/blended voices use included tensors where available, otherwise managed acquisition of only the requested voice. Voxtral requires the selected voice's local embedding before loading. Full model-family and portability acceptance gates remain tracked in [the implementation checklist](docs/implementation-checklist.md).

## CLI Usage

### Single file (default model/settings)

```bash
python Libro-tts.py input.txt -o tests_output/book_ch1
```

Default output is user-focused and concise (model/voice/speed/lang and save path), with an animated chunk progress bar during synthesis.
Use `--verbose` to enable full diagnostic logging.
Long text is automatically split into sentence-aware synthesis chunks targeting roughly 10-15 seconds of audio each, then reassembled into one contiguous output file.
If parallel synthesis fails, its partial audio is discarded and the entire text is retried serially. Chunk progress resets to zero for that retry and advances only after each chunk passes validation. A failed serial retry preserves any previously complete output.
The audio format determines the final extension (WAV by default). A matching output extension is preserved; otherwise the final suffix is replaced or appended. For example, `-o chapter.wav --audio-format mp3` writes `chapter.mp3`. Default input-derived names preserve dotted chapter stems, such as `chapter.1.txt` → `chapter.1.wav`.

Choose one input mode: a single file or `--input-dir`. Input types, output destinations, formats, and speeds are checked before model acquisition. Single-file decoding and empty-text checks also run before the backend preflight; each batch input is decoded and checked before its own generation, so a bad item can fail while valid items continue. Output destinations that would replace an input, including symlink aliases, are rejected.

Supported formats are `wav`, `mp3`, `flac`, `ogg`, `opus`, `vorbis`, `pcm`, and `raw` (case-insensitive). Compressed formats require the project-local FFmpeg, checked before acquisition. `pcm`/`raw` retain upstream headerless PCM behavior. Speed overrides must be finite and positive, except Spark accepts zero and rounds nonnegative values to its existing `{0.0, 0.5, 1.0, 1.5, 2.0}` categories. Values that overflow chunk sizing are rejected.

### Execution controls

```bash
bash run.sh input.txt -o output.wav --serial
bash run.sh input.txt -o output.wav --workers 2 --parallel-batch-size 6
```

The existing defaults remain two workers, parallel execution for at least three chunks, and a maximum of 12 chunks per pool batch. `--serial` and `--workers` are mutually exclusive; worker/batch counts must be positive integers. CSM/Dia retain their serial reliability guards. Successful parallel execution avoids loading an unused parent model; serial work/fallback loads it when needed. Each input retains its own pool, while a batch reuses successful reference transcript preparation when the model, store, audio, and sibling text are unchanged.

Smaller pool batches reduce retained WAV chunk data; they do not limit model memory. Warm serial execution can be faster for short inputs, while parallel work may help longer inputs at greater memory cost. See [the measured resource report](docs/performance-2026-10-03.md). Suppressed model output retains only an 8 KiB tail for failures; `--verbose` shows upstream diagnostics.

### Model-specific options

The pinned runtime does not provide uniform voice/speed controls:

| Family | Voice selection | Effect of `--speed` |
|---|---|---|
| Kokoro | Named/blended voice | Speech rate and chunk sizing |
| Qwen3 | Named speaker | Chunk sizing; upstream speed is unsupported |
| CSM / Dia | Reference WAV and transcript | Chunk sizing; upstream speech speed is ignored |
| Spark | Reference WAV and transcript by default | Chunk sizing when cloning; categorical speech speed only without a reference prompt |
| Chatterbox | Reference WAV | Chunk sizing; upstream speech speed is ignored |
| Soprano | Upstream voice option is ignored | Chunk sizing; upstream speech speed is ignored |
| Voxtral | Named embedding | Chunk sizing; upstream has no speech-rate control |

These are source-verified behaviors for the locked mlx-audio revision, not guarantees for later updates. `--list-models` reports catalog capabilities. Model path overrides remain an internal Python API; see [maintenance guidance](docs/maintenance.md).

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

Before processing any batch files, Libro-TTS calculates every destination and rejects collisions. For example, `chapter.txt` and `chapter.wav.txt` both map to `chapter.wav`. Leading-dot stripping, case/Unicode variants, and existing output symlinks pointing to the same file are also checked. Case and canonical Unicode variants are rejected even on a case-sensitive volume. The error lists conflicting inputs and destinations; rename those inputs or place them in separate input directories. A rejected batch creates no output directory or audio files.

### Output replacement and recovery

Successful generation deliberately overwrites an existing destination. Encoding first writes a unique hidden `.libro-audio-*.tmp.<format>` file beside the destination, preserving its audio extension. Only a successful nonempty encode followed by file synchronization is published with an atomic replacement. Encoding or publication failure preserves the previous output and removes that run's temporary file. A failed batch item is reported while other unambiguous inputs continue.

Existing output symlinks are followed and preserved. Replacement retains the existing file's permission mode; new files use the normal creation permissions/umask. Concurrent runs targeting the same destination use the last successful replacement.

WAV staging writes each accepted chunk as PCM16 instead of retaining the whole audio array. It preserves the pinned writer's clipping/quantization behavior. A serial retry truncates partial staged audio before restarting. WAV files exceeding the RIFF 4 GiB size limit fail clearly; split the input or choose another format. MP3/FLAC and other formats still retain the complete waveform before encoding.

Generated audio must contain finite real numeric samples with a positive integral sample rate and a consistent channel layout across segments and text chunks. Missing/empty stream results are skipped, but synthesis without any audio fails. Mono vectors and the existing single-row/single-column mono forms are accepted; other 2D audio uses channel-last layout. A `(1, 1)` result remains a one-sample mono waveform. Float16 samples are promoted to float32 before encoding so their fractional amplitudes are preserved. No automatic resampling, channel mixing, or amplitude adjustment is added by these checks.

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
- CLI inputs must be UTF-8; an initial UTF-8 BOM is accepted and removed before synthesis. Invalid UTF-8 produces a decoding error. The Python runtime API retains its configurable encoding order and historical Latin-1 fallback; fallback use is reported.
- Prompt-based model defaults (`csm`, `dia`, `spark`, `chatterbox`): `./reference_voices/Britney.wav`
- Named-voice models (`kokoro`, `qwen3_tts`, `voxtral_tts`) use upstream voice names via `--voice`, not `./reference_voices`.
- If you do not pass alternate `--model/--voice/--speed`, defaults are preserved.
- Prompt-based models accept `--voice` as either:
  - a direct `.wav` file path, or
  - a prompt name from `./reference_voices` (for example `--voice Britney`).
- For cloning models that use transcripts (`csm`, `dia`, `spark`), Libro-TTS also looks for a matching
  `.txt` file beside the selected `.wav` (for example `Britney.wav` + `Britney.txt`).
- Reference transcripts accept UTF-8 with or without a BOM, then Latin-1 for existing reference files. Resolution runs once per generation and shares normalized text across chunks; successful preparation is reused within a batch until the reference inputs change. Verbose logs identify explicit/file/STT provenance. Explicit nonempty runtime `ref_text` takes precedence over a sibling file.
- Missing or empty transcripts use the local managed Whisper model; the first online acquisition is about 1.6 GB. Offline transcription requires its complete local snapshot. Empty/invalid transcription results fail clearly. Spark may also transcribe an unusually long sibling transcript to recover prompt alignment, retaining its original trimmed file text if transcription fails or returns empty text.

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
bash scripts/check.sh
```

After development setup, this runs offline lock/dependency checks, the fast test suite, Ruff, shell syntax, and blocked-network command/import smoke. Use the same `UV_CACHE_DIR` as setup if overridden; offline lock verification requires warmed uv metadata. The pinned macOS CI workflow runs the same gate. See [maintenance and update procedures](docs/maintenance.md) and [the implementation checklist](docs/implementation-checklist.md) for remaining acceptance gates.

### Heavy real integration matrix (opt-in)

Uses `input.txt` and public-domain fixture text files to run generation across all supported models.

```bash
.venv/bin/python tests/integration_run_all_models.py --run
```

Optional second pass to verify offline local reuse:

```bash
.venv/bin/python tests/integration_run_all_models.py --run --offline-second-pass
```

## Notes

- Quality-first policy: model conversion/downloading is full precision only (no quantization).
- Batch generation favors output quality and consistency over low-latency streaming.
