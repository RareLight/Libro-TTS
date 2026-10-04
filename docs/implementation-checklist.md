# Libro-TTS implementation checklist

Based on [the October 2, 2026 assessment](quality-control-2026-10-02.md). Each checked item requires implementation and recorded validation; an unchecked acceptance gate remains open even when its supporting code exists.

Preserve the current CLI defaults, model quality policy, voice/reference files, and unrelated changes. Keep Python dependencies in a disposable project `.venv` and model assets in persistent `models/`. Changes to execution defaults require measurements. Heavy model tests and subjective listening remain separate from fast automated checks.

Scope update October 3, 2026: the user does not require full Whisper or Chatterbox model verification for this work and has deferred release portability. Those checks remain unverified and are excluded from the current output-safety acceptance gates. A later scope decision limits real-model acceptance to Kokoro and Soprano for now; Qwen3, CSM, Dia, Spark, and Voxtral generation/resource checks are deferred. On October 4, the user reopened technical portability checks for that selected scope; public-distribution and second-physical-Mac acceptance remain separate.

## 0. Baseline and delivery

- [x] Inspect project instructions, application, scripts, configuration, and tests.
- [x] Record baseline source revision and installed dependency versions in the assessment.
- [x] Reproduce the material correctness and isolation defects using disposable fixtures.
- [x] Draft this full checklist with implementation and acceptance gates.
- [x] Record implementation progress and verification below as each stage lands.
- [x] Review this delivery's diff for compatibility, unrelated changes, secrets, and accidental generated assets.
- [x] Record exact setup/run/validation commands and outstanding operator gates for this delivery.

## 1. Reproducible Python environment

- [x] Add project metadata with Python 3.12 and macOS arm64 runtime scope.
- [x] Declare direct runtime dependencies, TTS/STT extras, Voxtral tokenizer support, and the English spaCy pipeline.
- [x] Preserve the inspected core dependency versions for the initial migration.
- [x] Generate a resolved lockfile with distribution hashes for version control.
- [x] Add development dependencies separately from runtime dependencies.
- [x] Add an explicit setup command that installs only into project `.venv`.
- [x] Keep setup separate from generation; never install or upgrade packages while synthesizing.
- [x] Keep `.venv`, model downloads, caches, and generated output out of Git.
- [x] Validate a clean local environment against the lockfile and dependency consistency checks.
- [x] Verify Python package origins are inside `.venv`, with no user/base site-package exposure.
- [x] Record the initial Python patch version; assess future patch updates separately.

Acceptance: a fresh locked install requires no undocumented dependency additions and passes the fast suite without the shared Conda environment's packages.

## 2. Launch and diagnostic contract

- [x] Change `run.sh` to call project `.venv/bin/python` directly.
- [x] Preserve launch through symlinks, arguments with spaces, and the caller's working directory.
- [x] Fail with actionable setup instructions when `.venv` is absent.
- [x] Prevent inherited Python path/home settings from selecting external dependencies.
- [x] Require the project environment for direct generation, using interpreter prefix identity.
- [x] Keep help/list commands usable without importing the synthesis stack.
- [x] Make diagnostics report project environment identity, platform, dependency versions, cache roots, and encoder selection.
- [x] Return nonzero when required diagnostic checks fail.
- [x] Distinguish model-family detection from successful model loading and synthesis.
- [x] Update smoke, integration, benchmark, and export entry points to use the same contract.
- [x] Repair the direct integration runner's package-import failure.
- [x] Remove the smoke script's obsolete skip-preflight flag.
- [x] Update README and project instructions with tested commands.
- [x] Test launching outside the project and from an unrelated activated environment.
- [x] Test missing/wrong environment paths without model downloads.

## 3. Cache bootstrap and offline policy

- [x] Add a standard-library-only bootstrap before third-party imports.
- [x] Remove eager CLI/synthesis imports from package initialization.
- [x] Set absolute project paths for HF home, current/legacy hub cache, Xet cache, and asset cache.
- [x] Configure relevant supported transformer/dataset cache paths and tokenizer policy.
- [x] Set offline flags before imports and preflight subprocesses when requested.
- [x] Pass the same cache/offline policy to spawned workers and helper scripts.
- [x] Respect an explicitly inherited offline policy; avoid silently enabling network access.
- [x] Verify actual imported HF constants in a fresh process with conflicting inherited cache variables.
- [x] Test cache/offline inheritance with fresh subprocesses.
- [x] Restore mutated environment state in unit tests.
- [x] Prevent non-HF runtime package downloads by installing required spaCy assets during setup.
- [x] Ensure diagnostics do not download models or create unnecessary runtime state.
- [x] Add blocked-network offline tests for complete and incomplete auxiliary asset sets.

Acceptance: no supported entry point silently uses user-global caches, and offline execution never attempts acquisition through HF or non-HF paths.

## 4. Complete and portable model storage

- [x] Inventory primary and auxiliary assets for each pinned model/API combination.
- [x] Capture immutable revisions for new primary downloads and known historical snapshots; retain unknown provenance explicitly.
- [x] Extend store metadata with completion, revision, required assets, and relative paths.
- [x] Validate manifest schema and reject paths/symlinks outside the active store.
- [x] Migrate historical absolute-path entries using validated current-root candidates.
- [x] Preserve existing compatible model assets during migration.
- [x] Resolve Whisper transcription assets under project model storage.
- [x] Resolve CSM text tokenizer and Mimi codec locally.
- [x] Resolve Dia's audio codec locally.
- [x] Resolve Chatterbox's S3Tokenizer locally.
- [x] Add model-specific structural checks for bundled Qwen/Spark/Soprano tokenizer and codec files.
- [x] Validate Voxtral tokenizer and selected voice embeddings.
- [x] Resolve Kokoro named voices to available included local voice files.
- [x] Support existing accepted voice syntax deliberately, including any upstream blending syntax.
- [x] Download to a resumable staging location and publish only a validated complete primary snapshot.
- [x] Preserve partial progress and download metadata on interruption.
- [x] Reject config-plus-weights snapshots missing required primary tokenizer/voice assets.
- [x] Add a shared manifest transaction lock alongside per-model download locks.
- [x] Write manifest/config metadata atomically and surface actionable errors.
- [x] Test interrupted downloads, malformed manifests, missing primary assets, and concurrent different-model updates.
- [x] Test relocated model-store fixtures and default catalog resolution against copied Kokoro/Soprano trees with an empty HF cache.
- [x] Test relocated bundles with recreated environments, guarded original/global-cache reads, and offline generation. (Same-machine verification; a second physical Mac remains open.)
- [x] Confirm default Kokoro/Soprano acquisitions preserve unquantized catalog snapshots and synthesize successfully.
- [ ] Confirm real full-precision conversion behavior for a model requiring conversion. (Outside the current Kokoro/Soprano scope.)

## 5. Local native tools

- [x] Select a pinned, verified macOS arm64 FFmpeg distribution.
- [x] Install its binary inside the project environment and expose it to upstream encoding through local `PATH`.
- [x] Reject accidental fallback to external FFmpeg during generation.
- [x] Report the selected executable/version and document upstream license/provenance.
- [x] Keep base Python and macOS Metal/system frameworks documented as platform prerequisites.
- [x] Verify WAV, MP3, and FLAC generation with Homebrew absent from `PATH`.
- [x] Verify native encoder failure produces a useful error without damaging an existing output.
- [x] Validate portable export/recreation behavior and retention of the app license/present model cards.
- [ ] Complete model/dependency redistribution-license/source-notice review before public distribution.

## 6. Correctness and recovery

- [x] Precompute batch destinations and reject collisions before synthesis/output changes.
- [x] Cover audio-like stems, leading dots, dotted names, and case-insensitive filesystem collisions.
- [x] Write encoded audio to a sibling temporary file and atomically publish successful output.
- [x] Remove failed temporary files while preserving previously complete outputs.
- [x] Document the deliberate overwrite policy for successful generation.
- [x] Validate input mode, file/directory existence, nonempty text, and output destinations before acquisition.
- [x] Validate finite, model-appropriate speeds while preserving Spark's supported zero category.
- [x] Restrict audio formats to supported values and verify encoder availability early.
- [x] Decide and document output-suffix versus format precedence.
- [x] Decide and document UTF-8/BOM/fallback behavior consistently with the CLI.
- [x] Resolve reference transcripts once per generation, retaining file/explicit/STT provenance. (Batch context/cache remains in section 7.)
- [x] Restore Spark's long-sibling-transcript transcription recovery through the normal pipeline.
- [x] Cover missing/empty transcripts, long files, transcription failure, and prompt alignment recovery branches with mocked STT.
- [x] Define progress reset/recovery behavior after parallel failure.
- [x] Validate waveform/sample-rate/channel consistency and empty generation boundaries where warranted.

## 7. Resource use and abstractions

- [x] Expose serial/worker controls without changing established defaults prematurely.
- [x] Select execution policy before loading models; load the parent model only for serial work/fallback.
- [x] Add a run/batch reference context where it replaces repeated work or globals.
- [x] Measure pool startup and consider reusing a pool across compatible batch inputs. (Retain per-file pools; rationale in the resource report.)
- [x] Bound suppressed output capture while retaining useful failure diagnostics.
- [x] Combine duplicated isolated preflight work if the failure boundary remains clear.
- [x] Measure PCM memory and implement ordered incremental temporary-WAV output when justified.
- [x] Preserve sample order, channel shape, encoding behavior, and complete text coverage.
- [x] Move model capabilities/required kwargs/parallel restrictions into explicit catalog data or small adapters.
- [x] Extract runtime responsibilities only as needed by completed fixes.
- [x] Document model path overrides as an internal API or deliberately expose them.
- [x] Replace the deprecated Whisper loading API with a verified compatible API.
- [x] Narrow model-type metadata normalization and test each required compatibility adapter.
- [x] Document model-specific unsupported/ignored speed or voice options accurately.

## 8. Validation and maintenance

- [x] Expand fast tests for bootstrap, isolation, default catalog paths, storage, output atomicity, and reference provenance.
- [x] Exercise real multiprocessing orchestration, ordering, worker failure, and serial fallback with bounded fixtures.
- [x] Repair benchmark imports/lint and measure the actual generation/IPC/output path.
- [x] Record first-process/warm wall time, audio duration, memory, failures, and long-input behavior. (Disk caches were not cleared; large-family measurements remain open.)
- [x] Add CI for locked install, dependency consistency, fast tests, lint, shell syntax, and no-download command/import smoke. (Local gates/schema passed; hosted execution is open.)
- [x] Keep real Apple Silicon model tests opt-in or on an appropriate Mac runner.
- [x] Validate default Kokoro and Soprano locally without global cache/network.
- [ ] Benchmark a representative large family before generalizing small-model resource results or changing defaults. (Larger-family checks deferred by the user.)
- [ ] Execute the hosted CI workflow after delivery through a push/PR.
- [ ] Validate the remaining families with complete local auxiliary assets. (Whisper/Chatterbox excluded; other larger-family checks deferred by the user.)
- [x] Validate Kokoro/Soprano online first-run/offline reuse on disposable model roots. (Other families remain outside the current acceptance scope.)
- [x] Run long-chapter resource checks before changing parallel defaults.
- [x] Complete human listening acceptance for the supplied default Kokoro/Soprano samples. (Accepted October 4, 2026; other families and extended listening remain outside scope.)
- [x] Export the committed lock and relative asset metadata rather than freezing a shared environment.
- [x] Exclude known HF credential filenames, partial downloads, locks, and irrelevant cache metadata; validate included symlinks before export.
- [x] Recreate `.venv` at relocated destinations and verify selected-model offline execution.
- [x] Obtain user acceptance after copying the prepared package to a new location. (Confirmed October 4, 2026.)
- [ ] Confirm the bundle on a second physical Apple Silicon Mac before claiming cross-machine acceptance.
- [x] Document dependency/model revision updates and their regression gates.

## Progress and evidence

Implementation started October 2, 2026. The first delivery implemented the reproducible environment and entry-point contract, early cache/offline bootstrap, the local encoder, and included Kokoro voice resolution. Primary model-store work followed in the October 3 delivery below. Managed auxiliary acquisition followed in the next delivery below. Output recovery, measured WAV memory improvements, execution controls, and maintenance gates are now implemented. Full model-family acceptance and the explicitly deferred release work remain open. The historical assessment describes the pre-implementation state and is retained unchanged.

Verified first-delivery evidence:

- A fresh project `.venv` was created using Python 3.12.2, without shared/user site-packages. Required dependency origins were verified under `.venv`.
- The lock pins the existing mlx-audio Git revision `e42e1431fcf89af313375296c46d03a0153c4aa7`. PyPI's same-numbered release has different requirements. Clean synthesis also exposed spaCy's Click import dependency; the inspected compatible Typer 0.24.1/Click 8.2.1 combination is constrained in the lock.
- `uv pip check --python .venv/bin/python` passed for 115 installed distributions; `uv lock --check --offline` passed.
- `.venv/bin/python -B -m unittest discover -s tests -v`: 100 tests passed. New cases cover fresh-process import/cache ordering, inherited offline policy, environment identity/origins, launch symlinks/arguments, diagnostic failures, benchmark rejection before runtime imports, encoder selection, local/blended Kokoro assets, and export exclusions.
- Ruff for application/tests/scripts and Bash syntax checks passed. Help/list commands work without importing the synthesis stack. The direct integration runner reaches its opt-in skip path successfully.
- Outside the Metal-restricted sandbox, `bash run.sh --diag` passed using `.venv`, local caches, and local FFmpeg; all eight configured families were detected. This remains capability evidence, not proof of all-family synthesis.
- With network connections blocked, an isolated empty HF cache, Homebrew absent from `PATH`, and existing primary models selected through explicit local paths, Kokoro's default named voice and Soprano synthesized successfully. Their WAV/MP3/FLAC files were decoded/inspected: Kokoro 24,000 Hz/52,800 frames; Soprano 32,000 Hz/57,344 frames; all mono. Human listening was not performed.
- Portable-export fixtures preserve weights and the dependency/setup contract while excluding common HF credential files, incomplete files, and `.venv`. A frozen lock-derived requirements export succeeds without touching the user's uv cache. Full relocated model-store portability remains unverified.

Temporary logs: `/private/tmp/libro-implementation-local-tests.log`, `/private/tmp/libro-implementation-setup.log`, `/private/tmp/libro-implementation-diag.log`, `/private/tmp/libro-implementation-smoke.log`. Synthesis fixtures and outputs are under `/private/tmp/libro-implementation-smoke*`; temporary evidence may be removed by system cleanup.

Current commands:

```bash
bash scripts/setup.sh
bash run.sh --diag
bash run.sh input.txt -o output.wav --offline
.venv/bin/python -m unittest discover -s tests -v
uv pip check --python .venv/bin/python
uv lock --check --offline
.venv/bin/python -m ruff check libro_tts tests scripts Libro-tts.py
```

### October 3, 2026 delivery: primary model-store reliability

Implemented staged acquisition pinned to a resolved repository commit, complete remote file-size inventories, model-specific primary asset checks, nested weight integrity, shared atomic manifest transactions, and relative-path migration. Legacy assets remain reusable; known provenance is recorded without substituting current remote metadata for old files. Completion is scoped to the primary snapshot. See [model storage and asset coverage](model-storage.md) for the schema, recovery behavior, observed repo revisions, and separate auxiliary dependencies.

Verification:

- Full fast suite: 122 tests passed, including 22 new storage regressions. Cases cover interruptions after weights/before tokenizers, pinned resumable retry, preservation of historical partial progress, missing/truncated assets, malformed manifest/completion data, root containment, relocation after original-store removal, nested shards/indexes, failed atomic writes, inherited offline policy, and concurrent different-model processes.
- The local-input/separate-output conversion adapter was checked using an injected converter; full-precision flags are preserved. Real model conversion remains an acceptance gate.
- Real Kokoro/Soprano generation used default catalog paths in copied model trees, a migrated absolute manifest, an empty HF cache, blocked network connections, and one worker. Kokoro WAV: 24,000 Hz/79,200 frames; Soprano WAV: 32,000 Hz/83,968 frames; both mono. Source model assets and the live manifest were not migrated by these fixtures.
- Ruff and diff whitespace checks passed. Repository metadata was fetched without downloading weights. Real first-run downloads, remaining-family synthesis, and listening were not run.
- Logs/artifacts: `/private/tmp/libro-store-full-tests.log`, `/private/tmp/libro-store-real-smoke.log`, `/private/tmp/libro-store-repo-inventory.json`, and `/private/tmp/libro-store-real-smoke/`.

The next delivery implemented managed auxiliary acquisition and selected voice checks, as recorded below.


### October 3, 2026 delivery: managed auxiliary assets

Added `assets.py` for pinned selective acquisition of CSM text/Mimi, Dia DAC, Chatterbox S3Tokenizer, Whisper, and missing selected Kokoro voices. It reuses validated project HF-cache snapshots and records separate auxiliary completion inventories/relative manifest entries. HF loading and lazy synthesis are local-only after acquisition; fresh workers carry the active cache root. Runtime caches distinguish different stores. Voxtral validates the selected embedding before loading.

Reference transcript resolution moved out of the kwargs builder, restoring Spark's long-sibling-transcript recovery. Whisper now loads a managed local path with `mlx_audio.stt.load`; its cache distinguishes local paths. Explicit/file/STT branch behavior is retained inside preparation; persistent batch reference context/provenance and broader transcript failure coverage remain open.

Verification:

- Full fast suite: 142 tests passed, including 20 additional tests since the prior delivery. Cases cover declared auxiliary sets, pinned selective/resumable acquisition, failed repair preservation, malformed/truncated metadata, root containment, relocated blob symlinks, real HF lookups with blocked network, environment/constants restoration, local STT loading/cache identity, selected voice assets, Spark recovery through normal kwargs preparation, and a fresh spawned process using the caller's cache.
- Real CSM tokenizer lookup/encoding and Mimi/Dia codec loading/parameter evaluation passed using copied cached assets with network blocked. These checks do not claim CSM/Dia speech acceptance.
- Real first-run acquisition downloaded five CSM tokenizer JSON files and one Kokoro voice tensor into a disposable root. Offline reuse and real tokenizer encoding passed with connections blocked. No Llama language-model weights were downloaded.
- Real Kokoro/Soprano synthesis passed under local-only loading from copied primary trees. WAVs: 24,000 Hz/79,200 frames and 32,000 Hz/90,112 frames, respectively; mono. Original model metadata was not migrated by the fixtures.
- The supported STT loader import/signature was verified in `.venv` with network blocked. Remote metadata confirmed Whisper/S3Tokenizer's required file names. Full Whisper transcription, Chatterbox synthesis, remaining-family TTS, real conversion, and complete relocated release validation remain open; their large model weights were not acquired by these checks.
- Ruff and diff whitespace checks passed. Asset/runtime policy is documented in [model storage](model-storage.md), README, and project instructions.

Temporary evidence: `/private/tmp/libro-assets-full-tests.log`, `/private/tmp/libro-assets-aux-tests.log`, `/private/tmp/libro-assets-real-smoke.log`, `/private/tmp/libro-assets-online-smoke.log`, and the corresponding fixture directories.

The next delivery implemented output collision checks and atomic publication, as recorded below.

### October 3, 2026 delivery: output collisions and atomic audio writes

Batch processing now calculates all final paths before output-directory creation, input reading, model acquisition, or synthesis. Existing filename normalization is retained. Ambiguous batches fail as a whole with the conflicting input names and destinations; case-folded canonical Unicode paths and existing output symlink aliases are included. Unique dotted chapter names remain unchanged.

Audio encoding now uses an exclusively created hidden sibling file with the correct audio suffix. The existing pinned writers write that file in place; publication requires a nonempty encode and successful file synchronization/close, then uses `os.replace`. Normal encoding errors, interruption, synchronization failure, and replacement failure clean up the owned temporary file and preserve the previous audio. Output symlinks remain intact, existing permission modes are retained, and new files retain normal umask-based creation modes. Successful overwrite behavior and audio-format suffix precedence are unchanged and documented in README.

Verification:

- Full fast suite: 157 tests passed, including 15 new output-safety regressions. Cases cover normalized/audio-like/hidden/dotted names, case/Unicode collisions, all conflict groups, rejection before generation/output changes, CLI nonzero status, existing symlink aliases, staged writes across WAV/MP3/FLAC, partial encode/disk/interrupt failures, empty encoder results, sync/rename failures, file modes, symlink preservation, temporary-file ownership, and batch continuation after an encoding failure.
- Real encoding used a 0.25-second synthetic tone without TTS models or downloads, with network blocked and the project FFmpeg on `PATH`. WAV, MP3, and FLAC were encoded and decoded for mono and stereo; all six outputs were 24,000 Hz/6,000 frames with the expected channel count and nonzero waveform.
- A real FFmpeg invocation with an invalid sample rate failed with its encoder error. The previously encoded MP3 retained its identical SHA-256 digest, and no owned temporary files remained.
- Ruff and diff whitespace checks passed. Unrelated working-tree changes were preserved. Full Whisper/Chatterbox verification was not run, and release-portability work remains deferred as requested.

Temporary evidence: `/private/tmp/libro-output-focused-tests.log`, `/private/tmp/libro-output-full-tests.log`, `/private/tmp/libro-output-real-encoding.log`, and `/private/tmp/libro-output-real-encoding/`.

The next delivery implemented CLI/input/reference validation, as recorded below. Release portability and the excluded model-verification gates are not prerequisites for these deliveries.

### October 3, 2026 delivery: early request validation and reference transcripts

Added backend-free shared request validation. The CLI rejects conflicting input modes, missing/wrong input types, invalid formats, non-finite/invalid speeds, and unusable output destinations before backend work. Single-file text is decoded once before backend preflight and reused by generation. Empty text after sanitization is rejected before acquisition, including each batch item's generation. Batch destinations are checked together before creating the output directory; an output alias to any input is rejected using the same case/Unicode/symlink policy as collision planning. Compressed-format requests check the local encoder before acquisition.

CLI decoding remains strict UTF-8 with initial BOM removal. The Python API retains configurable encodings and its historical Latin-1 fallback. Reference siblings retain UTF-8/BOM then Latin-1 decoding. Empty reference files now trigger the missing-transcript fallback instead of a misleading decoding error. Resolution returns explicit/file/STT provenance, shares the prepared transcript across all chunks, and logs its source in verbose mode. Invalid/empty STT results fail clearly; failed or empty Spark recovery retains the original file transcript before trimming. Spark zero-category and normal nearest-category behavior are preserved, including internal model-path overrides.

Verification:

- Full fast suite: 189 tests passed, including 32 new request/reference regressions. Cases cover conflicting/missing inputs, wrong file types, invalid/overflowing speeds, Spark zero/rounding, unsupported and case-insensitive formats, encoder failure, empty/sanitized-empty inputs, unwritable/invalid outputs, protected input aliases across case/Unicode/symlinks, batch continuation, strict CLI decoding, library fallback, BOM removal, read errors, and single-file read reuse.
- Transcript tests cover explicit/file/STT precedence and provenance, missing/empty/BOM-only siblings, UTF-8/Latin-1 preservation, unreadable files, failed/empty/invalid STT results, Spark successful/failed long-file recovery and truncation, and one reference-file read shared across multiple synthesis chunks. STT and speech generation are mocked; no Whisper or Chatterbox model verification was performed.
- Ruff and diff whitespace checks passed. Generation defaults and successful overwrite behavior are preserved. No downloads or manual listening were required for this delivery.

Temporary evidence: `/private/tmp/libro-validation-full-tests.log`. Remaining active correctness work is parallel-fallback progress and waveform/sample-rate/channel consistency, followed by measured resource and maintenance work. Release portability remains deferred.

### October 3, 2026 delivery: parallel recovery and audio consistency

Parallel failure retains the established full-input serial retry, discards partial parallel audio, and now resets completed-chunk progress before restarting. Progress advances after sample-rate/channel checks accept the chunk and is closed on success or failure. A failed retry preserves the previously complete output through the existing atomic write boundary.

Waveform validation now rejects scalar/high-dimensional, non-real, and non-finite data; rate validation rejects nonpositive, fractional, non-finite, and nonnumeric metadata without truncation. Channel layouts must agree across a model's yielded segments and across serial/parallel text chunks. Empty/metadata-only results remain skippable, while wholly empty generation fails. Existing mono row/column conventions and valid stereo channel-last data remain supported. Single-sample `(1, 1)` mono output is preserved as a vector instead of squeezed into a scalar. Float16 is promoted to float32 because the pinned writer otherwise casts it directly to integer PCM and loses fractional amplitudes. Waveform scans happen at segment conversion; consistency checks between text chunks do not rescan the combined PCM.

Verification:

- Full fast suite: 211 tests passed, including 22 new audio/recovery regressions. Cases cover sample shapes/dtypes/finite values, float16 promotion, valid/deferred/missing/invalid rates, segment/chunk rate and channel mismatches, empty streams, sample and text ordering, partial parallel failure, progress reset/no-op behavior, full serial retries, and preservation of old outputs when retry results fail validation.
- Real spawned multiprocessing fixtures exercised the application pool with two workers, multiple batches, ordered assembly, worker exceptions, sample-rate/channel mismatches, and full serial fallback. Fixtures use no TTS models or Metal; model-family multiprocessing acceptance remains a separate gate.
- Real WAV encoding and PCM inspection passed for one-sample mono, 600-frame float16 mono, and 600-frame float16 stereo at 24,000 Hz. Decoded samples matched expected amplitudes within one int16 quantization step; no owned temporary files remained.
- Ruff and diff whitespace checks passed. No dependency pins or serial/parallel defaults changed, no model weights were acquired, and no manual listening was required for this delivery.

Temporary evidence: `/private/tmp/libro-audio-focused-tests.log`, `/private/tmp/libro-audio-full-tests.log`, `/private/tmp/libro-audio-real-encoding.log`, and `/private/tmp/libro-audio-consistency-mzosgh5k/`. Remaining work centers on measured resource use, execution controls, and ongoing maintenance/CI. Full Whisper/Chatterbox verification remains outside current scope, and release portability remains deferred.

### October 3, 2026 delivery: resources, capabilities, and maintenance

Implemented `--serial`, `--workers`, and `--parallel-batch-size` without changing worker/batch/chunk thresholds. Execution policy now precedes parent model loading. Successful parallel work loads models only in workers; serial work/fallback loads the parent when needed. A batch-scoped reference context reuses successful normalized transcripts and provenance, invalidating on changed model/store/offline policy, audio, sibling, or explicit text. Failures are not cached. Suppressed output retains an 8 KiB diagnostic tail. One isolated preflight now serves both backend and selected-family checks.

Catalog data now declares prompt/reference/parallel capabilities and source-verified voice/speed behavior for the pinned runtime. Metadata repair accepts canonical spelling, missing values, and only the known CSM/Spark/Voxtral adapters; unrelated architectures fail without rewriting. Internal repository overrides now acquire through the managed store and propagate offline failure rather than reaching the upstream loader remotely. The internal override/streaming contract is documented in [maintenance guidance](maintenance.md).

Measured full-file PCM/encoder overhead justified incremental WAV output. `audio_output.py` owns atomic transactions and ordered PCM16 staging, eliminating the combined waveform and full-file Python sample list. Serial retry truncates staged parallel audio; previous complete audio remains protected on failure. Unrecoverable storage/RIFF-size errors do not trigger redundant synthesis. Other formats retain their existing encoder. `diagnostics.py` provides bounded capture; broader runtime refactoring was unnecessary.

The rewritten benchmark times the real generation/pool/IPC/WAV path, records duration/RTF, phased timings, PCM sizes, parent high-water and sampled descendant RSS, failures, and bounded process cleanup. [The resource report](performance-2026-10-03.md) and [JSON evidence](performance-2026-10-03.json) retain the measurements. On a 96 MB/1,000-second synthetic fixture, sampled serial peak fell from 1,569.8 MiB buffered to 79.1 MiB incremental (incremental parent's high-water: 91.2 MiB). This is output-memory evidence, not model-speed evidence. Kokoro's full 40-chunk input produced 367.825 seconds of audio: 21.011 seconds serial, 18.833 parallel; sampled peaks 635.4/1,364.2 MiB. Short warm Kokoro/Soprano runs favored serial. Defaults and per-file pools remain unchanged; cross-file pool reuse is not justified by these limited measurements.

Added a pinned macOS CI workflow and `scripts/check.sh` for locked setup/dependency consistency, unit/mocked tests, Ruff, shell syntax, blocked-network commands/imports, and the heavy runner's opt-in boundary. Fast tests now supply disposable reference fixtures and do not depend on ignored private voice assets. [Maintenance procedures](maintenance.md) cover dependency/model updates, offline cache requirements, rollback, real-generation gates, and listening.

Verification:

- `UV_CACHE_DIR=/private/tmp/libro-uv-cache bash scripts/check.sh`: **239 tests passed**; 115 installed distributions are compatible; offline lock check resolved 116 packages. Ruff, shell syntax, fresh-process no-download help/model/voice/import checks, and opt-in integration skip passed. This uses the populated cache from earlier setup; a separate empty-cache offline lock failure was an expected missing-metadata condition, not an application defect.
- New tests cover positive execution controls, lazy parent loading/fallback, catalog policy, bounded diagnostic capture, batch reference reuse/invalidation/failure handling, architecture adapters, shared probing/errors, managed online/offline repo overrides, streamed/legacy PCM parity, ordered output/reset, disk failures/nonretryable storage errors, empty/layout/rate/RIFF failures, permissions/symlinks, fixture benchmarking, early empty-text rejection, and timeout/process ownership.
- Real four/five-chunk CLI-path Kokoro/Soprano checks passed serially and with two workers/batch size two, using copied local assets with internet sockets blocked in synthesis processes. Preflight and actual dispatch were exercised; only storage/cache destinations were redirected to disposable roots. All decoded outputs were mono PCM16 and nonzero: Kokoro both 24,000 Hz/901,800 frames; Soprano 32,000 Hz/1,052,672 serial and 1,062,912 parallel frames. Existing output placeholders were replaced successfully and no owned staging files remained. Stochastic duration differences are not speech-quality acceptance.
- CI YAML/job configuration parsed successfully; pinned action refs were verified. A hosted run was not triggered because no push/PR was requested. Diff whitespace checks passed. Dependency/model defaults and precision remain unchanged; no new weights were acquired for resource or final synthesis checks.

Temporary logs: `/private/tmp/libro-maintenance-final-check.log`, `/private/tmp/libro-resource-last-focused.log`, `/private/tmp/libro-resource-final-smoke.log`; final smoke audio/report under `/private/tmp/libro-resource-final-smoke/`. Benchmark metrics are retained in the repository JSON because temporary logs may be cleaned up.

Remaining acceptance requires listening, a hosted CI run following delivery, and explicit opt-in large-family acquisition/generation/real-conversion checks. Only Kokoro/Soprano primary weights are currently present in the project. Full Whisper/Chatterbox verification remains excluded and release relocation/distribution remains deferred by the user. No further mandatory manual step is needed to validate the mechanical changes covered by these automated gates; those results do not close the separate listening or all-family acceptance gates.

### October 3, 2026 acceptance follow-up: fresh primary acquisition

The user selected Kokoro/Soprano only for the remaining real-model work; larger-family acquisition/generation/resource tests are deferred along with the earlier exclusions. No application changes were necessary for this acceptance check.

Both primary models were acquired through the normal managed store from genuinely empty disposable model/HF-cache roots. Before downloading, offline resolution correctly rejected each empty root. Kokoro downloaded all 122 remote inventory files (389,411,255 bytes) at commit `a71e4d38b236d968966a2002c4c895dbd12b1c3c`; Soprano downloaded all nine files (283,883,377 bytes) at `745350c27f356c3910eebcb49e29760dcf6a643c`. Publication produced downloaded completion inventories and relative version-2 manifest entries. The unquantized catalog snapshots were preserved; neither acquisition required conversion. Live project model directories/caches were untouched.

Fresh offline processes reused each newly downloaded model, with internet socket connections blocked in both the parent and spawned workers. Default generation passed serially and with two workers/batch size two. All WAVs decoded to nonzero mono PCM16, and no owned audio staging files remained. Kokoro produced 901,800 frames at 24,000 Hz in both modes (four chunks); Soprano produced 1,075,200 serial and 1,048,576 parallel frames at 32,000 Hz (five chunks). Soprano durations are stochastic. Model/revision/completion selection remained unchanged; the store's existing `updated_at` refresh is intentionally allowed. An initial helper assertion incorrectly required byte-identical manifests and was corrected after confirming only the expected timestamp changed; both synthesis modes had already passed.

[The retained acquisition report](acquisition-2026-10-03.json) records online/offline results and the ignored project-local listening sample directory. The samples include matching input text and default parallel outputs. Listening remains a human gate; no manual step is needed for the automated acquisition/cache/output checks. The prior 239-test maintenance result remains valid because this follow-up changes documentation/evidence only. Hosted CI still awaits a requested push/PR, and real conversion/release acceptance remain outside the selected scope.

Temporary evidence: `/private/tmp/libro-primary-acquisition-online.log`, `/private/tmp/libro-primary-acquisition-offline.log`, `/private/tmp/libro-soprano-primary-acquisition-online.log`, `/private/tmp/libro-soprano-primary-acquisition-offline.log`. Disposable downloaded stores remain under `/private/tmp/libro-primary-acquisition-20261003/` and `/private/tmp/libro-soprano-primary-acquisition-20261003/`.

### October 4, 2026 delivery acceptance and commit

The user listened to both supplied default parallel samples and confirmed that Kokoro and Soprano work as expected. Kokoro sounded slightly more natural to the user, reinforcing the existing default; no catalog default was changed. This closes the selected samples' human listening gate, alongside the already recorded clean-acquisition/offline serial/parallel and 239-test maintenance evidence. Larger-family verification, actual conversion for families requiring it, and release portability remain deferred/outside the selected scope. Hosted CI awaits publication through a push/PR.

The QC implementation, tests, lock, workflow, documentation, and retained objective/subjective acceptance evidence are included in the delivery commit. Unrelated `.DS_Store` and pre-existing `.gitignore` changes are left outside that commit. Downloaded models, caches, private references, the virtual environment, and listening audio remain untracked.

### October 4, 2026 portability verification

The user reopened portability acceptance for the selected Kokoro/Soprano scope. Fixed export gaps: absolute legacy manifest entries/stale absent families, unchecked external/dangling symlinks, recursive destinations, incomplete final folders after failures, optional source-only asset folders, and exported cache logs/download metadata. Internal symlinks remain relative, complete primary inventories retain known revisions, and migration occurs only in the copied store. Existing source manifest/config bytes were verified unchanged. The app license, existing model cards/notices, setup/lock, and source-reference documentation are retained; `.venv` is recreated rather than copied.

Fresh managed Python/dependency setup succeeded with a dedicated empty uv cache. The fixed bundle was moved before setup into a spaced path, then moved again after setup and its disposable `.venv` recreated offline from that warmed cache. Both environments used Python 3.12.13 and 114 runtime-only distributions entirely inside the new prefixes. Internet socket guards and Python filesystem audit guards rejected original/global-cache reads; Homebrew was excluded from runtime PATH. Negative probes confirmed guards in fresh processes, and spawn/preflight inherited them.

Diagnostics, listing, Kokoro/Soprano serial/parallel WAV generation, and local MP3/FLAC encoding/decoding passed. All waveforms were finite/nonzero and no owned staging files remained. Nine internal cache links were checked after relocation. A second move/environment recreation passed diagnostics and Kokoro WAV/MP3 again. Full maintenance gate: 245 tests passed, dependency checks/Ruff/shell/no-download checks passed. Seven exporter tests cover the corrected boundaries. No larger-family generation or subjective reassessment was added.

See [the portability report](portability-2026-10-04.md) and [retained JSON evidence](portability-2026-10-04.json). This establishes guarded same-machine technical transfer, not execution on a second physical Mac or public redistribution readiness. Those remain explicit gates; unrelated model-family/conversion deferrals remain unchanged.

The user subsequently copied the prepared package to a new location, tested it, and confirmed that everything worked. This closes location-transfer acceptance. A different physical machine was not specified in that report.
