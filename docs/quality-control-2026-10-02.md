# Libro-TTS quality-control assessment and environment implementation plan

Assessment date: October 2, 2026. Source revision: `4c8087e`.

The application has a sensible foundation for a batch TTS CLI. Its separation of CLI, model catalog, model storage, environment checks, and generation is useful; the defaults, subprocess MLX preflight, and batch failure reporting should be preserved. A rewrite is not warranted. The main weaknesses are environment reproducibility, incomplete control of auxiliary downloads, storage recovery, and resource use during long or parallel generation.

The new requirement is **not fully met**. Python packages currently live in a machine-local, shared Conda environment rather than a project-local virtual environment. Primary TTS snapshots are project-local, but auxiliary asset resolution can use global caches or the network. MP3/FLAC encoding also depends on an external FFmpeg executable.

This assessment adds documentation only. No application code, installed packages, model configurations, or model manifest were changed. Focused reproductions used disposable fixtures. Real generation used existing models, blocked network connections, and wrote only under `/private/tmp`.

## Verified environment and dependency isolation

| Component | Observed configuration | Meets project-local isolation? |
| --- | --- | --- |
| Launcher | `run.sh:17–24` hard-codes `/Users/anna/miniconda3/bin/conda`, environment `tts` | No |
| Python | 3.12.2; executable and prefix under `/Users/anna/miniconda3/envs/tts` | Isolated Conda environment, but shared and outside this project |
| Project environment | No `.venv`, `pyproject.toml`, requirements file, or dependency lockfile | No reproducible local setup |
| Core packages | `mlx-audio 0.4.3`, `mlx 0.30.6`, `mlx-lm 0.31.1`, `huggingface-hub 1.8.0`, `transformers 5.4.0`, `numpy 2.2.6` | Installed in the shared `tts` environment |
| Additional packages | `misaki 0.9.4`, `spacy 3.8.7`, `en-core-web-sm 3.8.0`, `mistral-common 1.11.0`, `miniaudio 1.61` | Installed in the same shared environment |
| Primary TTS weights | `models/<key>/<repo>/`; explicit `snapshot_download(local_dir=...)` | Yes, except existing manifest paths can select another checkout |
| Hugging Face caches | App intends `models/.hf`; parent-process HF constants remained `/Users/anna/.cache/huggingface/{hub,xet}` | No reliable enforcement |
| Auxiliary models and voices | Whisper, codecs, tokenizers, and upstream Kokoro voice resolution | Not comprehensively managed |
| Native encoding | Installed `mlx_audio.audio_io` locates FFmpeg through `PATH`; current executable is `/opt/homebrew/bin/ffmpeg` | No |

A virtual environment isolates Python packages; it does not automatically redirect model caches or contain arbitrary external executables. Keep model data in `models/`, separate from the disposable `.venv`. Python documents that virtual environments should be recreated when moved, rather than copied: [Python venv documentation](https://docs.python.org/3/library/venv.html).

The installed `mlx-audio` metadata separates TTS and STT extras. A fresh installation must explicitly include the dependencies needed for both TTS and reference-audio transcription, plus the Voxtral tokenizer dependency. Installing just bare `mlx-audio` is insufficient as a complete application specification. The README currently mentions a `requirements.txt` that is absent from this checkout.

Only Kokoro and Soprano passed the application's current readiness check in the on-disk catalog inventory. Other manifest entries refer to missing snapshots; the Kokoro manifest path also refers to an old checkout location. These are observations about current local assets, not proof that those model families are broken.

## Findings requiring correction

Priorities below are remediation priorities: high for the new isolation/offline contract or demonstrated output loss, medium for recovery, portability, and maintenance defects. Reproduced means a focused fixture or real local synthesis exercised the behavior; inspection means the risk follows from the source but was not measured end to end.

### 1. High: cache configuration occurs after dependency imports

**Evidence:** `libro_tts/__init__.py:3`, `cli.py:22–29`, `runtime.py:26`, `store.py:13`, and `cli.py:174–179`.

Importing the package loads the CLI and model store, which imports Hugging Face before `configure_local_cache_environment()` runs. Preflight subprocesses also run before those variables are set. In a fresh process, importing the CLI established HF cache constants under the global user cache. Calling Libro-TTS's cache configurator changed `os.environ`, but the imported hub and Xet constants stayed global. This was reproduced without downloading anything.

The configurator sets the legacy `HUGGINGFACE_HUB_CACHE`, but not the current `HF_HUB_CACHE`; an inherited current variable can also defeat the intended project cache. Hugging Face documents that these settings are read at import time and that the current variable supersedes the legacy name: [HF environment variable documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

**Correction:** establish the cache and offline policy before any third-party import, including diagnostics, workers, and helper scripts. Set current hub, Xet, and asset cache paths explicitly. Remove the eager CLI import from package initialization or make it lazy. Add a subprocess test that inspects actual dependency constants, rather than only environment strings.

### 2. High: `--offline` does not cover all asset acquisition

**Evidence:** `cli.py:81–85`, `runtime.py:367–427`, and `store.py:59–73`.

The offline option guards the primary `ModelStore` download. It is not passed to automatic reference transcription: a fixture with `RuntimeOptions.offline=True` still called Whisper `from_pretrained(path_or_hf_repo="mlx-community/whisper-large-v3-turbo-asr-fp16")`.

Installed `mlx-audio 0.4.3` also resolves auxiliary assets outside this store: Chatterbox's S3Tokenizer, Dia's audio codec, CSM's text tokenizer/Mimi codec, and Kokoro named voices. These upstream loaders can access Hugging Face even when the primary model was loaded from a local directory.

Kokoro supplied a concrete portability failure. With an empty isolated HF cache, existing local weights, and existing `voices/af_aoede.safetensors`, normal named-voice generation failed with `LocalEntryNotFoundError`. Passing that same local voice file directly succeeded. Its upstream pipeline requests a separate HF snapshot for a named voice rather than selecting the included local voice file. This confirms that copying the primary snapshot alone is not sufficient for the default model's current named-voice path.

Misaki additionally calls `spacy.cli.download()` when its English pipeline package is missing. That operation uses network requests and invokes `sys.executable -m pip install`; HF offline flags do not block it. The package is installed in today's Conda environment, so this branch was inspected, not executed.

**Correction:** bootstrap HF/transformers offline settings before imports; make the application explicitly resolve required auxiliary assets; select available local Kokoro voice files through a small adapter; preinstall and pin the spaCy package. Missing assets in offline mode should produce a specific local error. Test offline operation with blocked network access and no global cache, including missing-asset cases.

### 3. High: batch output collisions silently discard results

**Evidence:** `runtime.py:715–739` and `1041–1085`.

`chapter.txt` and `chapter.wav.txt` both normalize to `chapter.wav`. A fixture batch reported `files_processed=2`, but produced one output containing the second input's result. Leading-dot stripping creates further collisions. There is no preflight check for duplicates.

**Correction:** calculate all output paths before synthesis and reject ambiguous batches, or apply a documented deterministic naming policy. Account for case-insensitive filesystems. Add tests for audio-like stems, leading dots, dotted names, and collisions on the supported filesystem behavior.

### 4. Medium: failed downloads can be accepted as complete on retry

**Evidence:** `store.py:64–67`, `126–134`, `161–198`, and `env.py:269–275`.

Readiness requires `config.json` and weight files, with useful shard checks, but not tokenizer/voice/codec assets or successful snapshot completion. A fake Voxtral download wrote config and weights, then failed before `tekken.json`. The next `ensure_model()` reused that incomplete directory without another download. The later tokenizer preflight's suggestion to rerun for a download therefore cannot repair this case.

Deleting incomplete directories before retry also discards downloadable partial progress.

**Correction:** download to a staging location, validate the complete required asset set, record successful completion/revision, and publish the snapshot only after success. Preserve resumable download metadata where possible. Test interruptions after weights but before required auxiliary files. Avoid expensive full weight hashing on every normal launch.

### 5. Medium: the shared manifest has no shared transaction lock

**Evidence:** `store.py:62–63` and `283–310`.

Locks are per model key, while every key updates the same JSON file. Two different model downloads can read the same manifest and overwrite one another's entries. A synchronized two-writer fixture preserved only one of two entries. Direct truncating writes can also expose partial JSON to readers or after interruption.

**Correction:** retain per-model download locks and add a separate short manifest lock around read/update/write. Write a temporary file in the manifest directory and atomically replace the manifest. Validate its nested schema. Test two different model keys concurrently and interrupted writes.

### 6. Medium: absolute manifest paths undermine project-local portability

**Evidence:** `store.py:113–116` and `298–305`.

The manifest accepts an existing absolute path without constraining it to the active store. A copied-store fixture loaded a model from the original store outside its own root. The live Kokoro entry is already an absolute path to a different checkout, although that path currently does not exist and the local candidate fallback can recover it.

**Correction:** store paths relative to the model root, resolve them against the active root, and reject paths or symlinks that escape it. Migrate old manifests by matching allowed local candidates. Prefer the current project's own snapshots over external historical paths.

### 7. Medium: output writes can replace a good file with a partial file

**Evidence:** `runtime.py:786–798` and the installed encoder's overwrite behavior.

Audio is written directly to the final destination. A fixture encoder wrote partial content and then raised an error; the previous complete output was lost and the partial file remained.

**Correction:** write to a sibling temporary file, check successful encoding, then atomically replace the destination. Clean up failed temporary files. Preserve the existing overwrite policy for completed conversions and document it clearly.

### 8. Medium: validation is late and allows invalid runtime values

**Evidence:** `cli.py:38–84`, `174–196`, `runtime.py:145–151`, and `cli.py:274`.

The parser accepts non-finite speeds and unrestricted format strings. `--speed inf` reaches an integer conversion that raises `OverflowError`, outside the CLI's caught exception types. Negative speeds are also accepted for models where they are invalid; Spark needs its separate categorical policy, which legitimately includes zero. Unsupported encoding formats and missing/empty inputs can be discovered after expensive checks or model loading.

Single-file and directory inputs can both be specified; directory mode silently wins. Explicit output suffixes are replaced according to `--audio-format`, while the README says extensions are used directly. The runtime supports encoding fallbacks, but the CLI explicitly disables them; the project instructions and executable contract disagree.

**Correction:** validate finite, model-appropriate speed, supported formats, input mode, input files, and intended output paths before model acquisition. Decide and document one suffix/format precedence rule and one text-encoding policy. Preserve current behavior deliberately where compatible; do not silently change voice or encoding defaults.

### 9. Medium: Spark's long-file transcription recovery is bypassed

**Evidence:** `runtime.py:349–352`, `406–414`, and `434–458`.

The kwargs builder reads the sibling transcript. The later preparation step marks an existing `ref_text` as originating from kwargs, while the long-file transcription branch requires source `file`. A fixture with a 960-character sibling transcript passed through the normal builder and preparer without calling transcription; it was trimmed to 311 characters instead. The special recovery path is unreachable through that normal path.

**Correction:** resolve reference text once and retain its provenance. Test the complete builder-to-preparer sequence, including long sibling transcripts and transcription failure. Preserve alignment between reference audio and the resolved transcript.

### 10. Medium: maintenance commands and tests leave material gaps

**Evidence:** `tests/integration_run_all_models.py:16`, `scripts/smoke_all_models.sh:55–56`, `tests/test_env_cache.py:10–18`, and `tests/test_mocked_pipeline.py:421–435`.

The documented direct integration command fails immediately with `ModuleNotFoundError: No module named 'libro_tts'`, even without `--run`. The smoke script's `SKIP_PREFLIGHT=1` option adds a deleted CLI flag, `--skip-mlx-preflight`, which the parser rejects.

Mocked pipeline cases force one worker and usually use model path overrides, bypassing catalog defaults. Fake models accept arbitrary kwargs. The cache test checks directories and environment strings, missing the actual import-time cache behavior; it also leaves environment mutations in its process. Real multiprocessing orchestration, fallback behavior, and offline auxiliary asset acquisition lack adequate coverage.

**Correction:** use a tested module/package invocation or correct the script import path, update stale smoke options, restore environment state in tests, and add focused boundary tests. Keep heavy model runs opt-in. Test the actual default catalog route as well as explicit model overrides.

### 11. Medium: diagnostics report failure with exit status zero

**Evidence:** `cli.py:127–159`.

The sandboxed runtime capability probe aborted with exit code `-6` during Metal initialization. `--diag` printed `status: error` and unknown model support, but returned zero. The same read-only probe succeeded outside the sandbox, so that abort is attributable to this execution context rather than evidence of a broken normal installation.

**Correction:** expose an actionable diagnostic status and return nonzero for failed required checks, with tests. Distinguish installed family names from successful model loading and synthesis; detecting a family directory is not an end-to-end compatibility guarantee.

## Efficiency and abstraction recommendations

These are opportunities supported by source inspection. No speedup or memory saving was benchmarked during this assessment.

| Opportunity | Evidence and likely cost | Bounded improvement |
| --- | --- | --- |
| Redundant model loading in parallel mode | Parent loads at `runtime.py:894`; each spawned worker loads again at `615–620`. Default two workers can retain three TTS instances, plus optional STT | Expose worker count/serial mode; select execution policy before loading; load the parent only if needed for serial/fallback |
| Repeated worker startup in batches | A new pool is created inside every file's `_generate_audio_parallel()` | Consider a pool scoped to one batch/model after measuring startup versus synthesis cost |
| Chapter-size memory growth | All audio chunks are retained; final `np.concatenate` allocates another chapter-size buffer; encoding can make more PCM/byte copies | Write ordered chunks to a temporary WAV incrementally, then encode once and publish atomically |
| Unbounded suppressed logs | Each quiet generation captures stdout/stderr in `StringIO` | Use an appropriate sink or bounded capture; retain useful error diagnostics |
| Duplicate preflight imports | MLX import preflight and capability probe initialize separate interpreters | Combine into one isolated probe that returns capabilities and failures |
| Repeated reference work | Builder/preparer overlap; absent sibling transcripts can trigger transcription for every chunked input file | Prepare one reference context per run/batch and reuse it |
| Misleading parallel benchmark boundary | `benchmark_chunk_parallel.py:105–153` excludes model/pool startup; workers return metrics rather than real audio arrays | Measure cold and warm total wall time, audio duration, memory, IPC, and failures using the actual orchestration |

Keep the existing module boundaries. Split the 1,091-line runtime gradually into text preparation/chunking, model/reference preparation, execution, and output writing when correcting the relevant defects. Put prompt requirements, accepted kwargs, parallel restrictions, and required auxiliary assets into the model catalog or small explicit adapters. The current scattered model sets duplicate those capabilities across runtime and scripts.

Use a run-scoped session for model/reference/pool lifetime only where it replaces today's mutable global caches. Avoid a generic backend framework, plugin architecture, database, or asynchronous scheduler for this CLI. Model path override support is actively tested despite being absent from the current CLI; classify it as an internal API or expose it deliberately before removing it as supposedly dead code.

Pin upstream compatibility deliberately. The current code uses the deprecated Whisper `Model.from_pretrained()` API, and model repos are named but not pinned to commits. Broadly rewriting `config.json` model types and swallowing metadata-write errors also deserve narrow adapter tests. Qwen's installed API documents speed as not directly supported, and Soprano accepts unused voice kwargs; model-specific behavior should be described accurately instead of implying uniform capabilities.

## Proposed implementation plan

This is a draft plan; the files and behavior below are proposed, not implemented.

1. **Establish the reproducible dependency baseline.** Add a minimal `pyproject.toml` and committed lockfile, preferably using the available `uv` tool with `.venv` as the project environment. Declare direct dependencies and the needed TTS/STT extras, Voxtral tokenizer support, and the compatible English spaCy pipeline. Start with the inspected working versions, then resolve and validate a clean macOS arm64/Python 3.12 environment. Keep the Python patch-level update separate from the functional migration. Treat the shared Conda environment's full package inventory as evidence, not as the application's dependency specification. No runtime package installation or automatic upgrade.

   Acceptance: fresh locked installation passes `pip check`, fast tests, and dependency-origin checks without user/base site-packages. Recreating `.venv` requires no undocumented manual package additions.

2. **Route every executable path through the local environment.** Update `run.sh` to execute `$PROJECT_ROOT/.venv/bin/python` directly and fail with setup instructions if missing. Retain symlink and caller-path behavior. Update smoke, integration, benchmark, and exporter paths to use the same environment. Have generation validate `sys.prefix` against the project `.venv`, rather than trusting `CONDA_DEFAULT_ENV` or `VIRTUAL_ENV`; resolving the Python executable symlink is not a reliable venv identity test. Preserve help/list/diagnostic usability where feasible with lazy imports. Add platform/architecture and encoder diagnostics.

   Acceptance: launching from another directory or an activated unrelated environment still runs project Python, uses project dependencies, and preserves argument handling. Wrong-environment direct generation fails clearly before downloads.

3. **Bootstrap cache and offline policy before imports.** Add a small standard-library-only bootstrap used by supported entry points. Set absolute project paths for `HF_HOME`, `HF_HUB_CACHE`, legacy hub compatibility if needed, `HF_XET_CACHE`, `HF_ASSETS_CACHE`, and relevant supported downstream caches. Apply offline settings before preflight/imports, and pass the same policy to workers. Verify actual library constants in a fresh subprocess. Keep credentials out of exported cache data; a project HF home can contain authentication files as well as cached assets, and the exporter currently copies the whole models tree.

   Acceptance: parent, preflight, spawned workers, diagnostics, and helper scripts report local cache roots even when the caller supplied global cache variables. A run never silently falls back to global caches. Offline missing-asset tests make no network attempts, including non-HF download paths.

4. **Make model acquisition complete, local, and portable.** Extend store metadata to record repo commit/revision, completion, required auxiliary assets, and relative paths. Preserve existing model files; migrate or rebuild manifest metadata after validation. Resolve Whisper and each model's codec/tokenizer/voice dependencies within project storage. Map existing Kokoro named voices to included local files where supported. Add staging/completion validation, resumable recovery, a shared atomic manifest transaction, and root containment checks. Retain full-precision policy.

   Acceptance: an isolated project copy synthesizes offline without the original checkout or user cache. Missing tokenizer/codec/voice files are diagnosed and repaired online; interrupted downloads are not accepted as ready. Concurrent model acquisition preserves all entries.

5. **Include native runtime tools needed by the stated formats.** If all non-OS runtime dependencies must be project-local, provide a pinned macOS arm64 FFmpeg build under a project tools directory and prepend that directory only for Libro-TTS's execution. Verify provenance, redistribution license, and version. Do not rely on Homebrew's mutable `PATH` entry. The base Python installation and macOS Metal/system libraries remain platform prerequisites; `.venv` is not a container for the operating system.

   Acceptance: WAV, MP3, and FLAC checks pass with Homebrew removed from `PATH`; diagnostics identify the selected local encoder. If external FFmpeg is intentionally retained, explicitly record that narrower dependency-isolation contract.

6. **Correct demonstrated data/recovery defects.** Add collision preflight, atomic output publication, early argument validation, reference-context provenance, and meaningful diagnostic exit status. Repair the integration and smoke commands. Add the focused regression cases described above before broader runtime extraction.

   Acceptance: colliding batches fail before output changes; failed encoding preserves the previous complete output; invalid arguments fail before model acquisition; Spark long-file recovery is exercised through the actual pipeline.

7. **Measure and then optimize.** Expose execution controls while retaining compatibility. Benchmark actual serial versus parallel cold/warm runs on representative small and large models, including total memory and a long chapter. Then remove redundant parent loading, reuse reference preparation/pools if beneficial, and bound PCM memory. Keep any default change separate and evidence-backed.

   Acceptance: documented wall-time/memory/failure evidence justifies each performance change; audio order, complete text coverage, format, sample rate, and voice behavior remain correct. Human listening remains a separate quality gate.

For ongoing maintenance, run locked installs, fast tests, narrow linting, shell syntax checks, and a no-download import/command smoke in CI. Keep Apple Silicon generation tests on an appropriate Mac runner or as documented operator gates. Update dependencies in small reviewed batches with the offline/auxiliary-asset matrix, record compatible package/model revisions, and maintain one tested setup/run contract across README, project instructions, and scripts. Portable exports should carry that lock and relative asset metadata rather than freezing whichever shared environment happened to invoke the exporter; recreate `.venv` at the destination.

## Validation performed and limits

| Check | Result |
| --- | --- |
| `/Users/anna/miniconda3/envs/tts/bin/python -B -m unittest discover -s tests -v` | 79 passed |
| Documented individual CLI test module | 8 passed; subsidiary invocation confirmed |
| Installed-environment `python -m pip check` | No broken requirements |
| Ruff on application, tests, and entry script | Passed |
| Ruff including scripts | Five E402 findings in benchmark imports following its path bootstrap; maintenance lint issue |
| Bash syntax on launcher, installer, and smoke scripts | Passed |
| Documented direct integration invocation without `--run` | Failed with package import error before reaching opt-in handling |
| Smoke script's removed skip-preflight flag | Parser rejected it, exit 2 |
| Sandboxed `bash run.sh --diag` | Probe aborted during Metal initialization; CLI exit 0 |
| Same read-only diagnostic outside sandbox, HF offline enabled | Passed; all eight configured families detected |
| Real Soprano, short text, empty isolated HF cache and network blocked | Passed: mono PCM16 WAV, 32,000 Hz, 57,344 frames, 1.792 seconds |
| Real Kokoro with default named voice, same isolated conditions | Failed: upstream snapshot absent from HF cache despite local voice assets |
| Real Kokoro using its existing local `af_aoede.safetensors` file | Passed: mono PCM16 WAV, 24,000 Hz, 52,800 frames, 2.2 seconds |
| Focused disposable fixtures | Reproduced late cache constants, offline auxiliary lookup, batch collision, incomplete snapshot reuse, lost manifest update, external manifest path selection, partial output replacement, infinite-speed exception, and bypassed Spark recovery |

The real synthesis checks exercised the runtime with local path overrides, explicit catalog settings, and one worker. They did not exercise a complete CLI acquisition run or the real parallel pool. No new model downloads, package installation, full all-model synthesis matrix, long-input memory benchmark, MP3/FLAC generation, subjective listening evaluation, or clean `.venv` installation was performed. Those remain implementation acceptance gates; family detection and mocked tests do not establish them.

Temporary evidence remains in `/private/tmp/libro-qc-unittest.log`, `/private/tmp/libro-qc-cli-tests.log`, `/private/tmp/libro-qc-diag.log`, `/private/tmp/libro-qc-diag-unsandboxed.log`, `/private/tmp/libro-qc-smoke.log`, `/private/tmp/libro-qc-kokoro-local-voice.log`, and `/private/tmp/libro-qc-smoke-2026-10-02/`. Real-smoke fixtures are `/private/tmp/libro-qc-smoke.py` and `/private/tmp/libro-qc-kokoro-local-voice.py`. Temporary files may be removed by normal system cleanup.
