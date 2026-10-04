# Maintenance and regression gates

## Routine changes

Run from the project root on an Apple Silicon Mac:

```bash
bash scripts/setup.sh --dev
bash scripts/check.sh
```

Setup installs `uv.lock` into `.venv` and exposes the wheel's local FFmpeg. It may download Python/packages, but generation never installs packages. Keep `models/` and private `reference_voices/` separate from the disposable environment. Recreating `.venv` must not remove those assets.

The check script verifies the offline lock, installed dependency consistency, the full fast unit/mocked suite, Ruff, shell syntax, and fresh-process help/list/import commands with internet connections blocked. It also verifies that the heavy integration runner skips without `--run`. It does not run large speech models. Fast tests provide disposable reference fixtures instead of requiring ignored/private voice files.

The offline lock check requires the uv metadata cache warmed by setup. If setup used an explicit `UV_CACHE_DIR`, use the same value for checks. An empty uv cache can make `uv lock --check --offline` fail even with a valid lock and installed packages; rerun the explicit setup/online metadata check rather than changing the lock to fix a cache miss. A runtime-only `--no-dev` install needs development setup before Ruff can run.

[The CI workflow](../.github/workflows/ci.yml) performs locked setup and these checks on `macos-15`. Actions and uv are pinned; checkout credentials are not persisted. Real-model work remains opt-in. Local gates and workflow parsing were verified; the hosted job requires a subsequent push/PR to execute. Runner support is documented by [GitHub](https://docs.github.com/en/actions/reference/runners/github-hosted-runners); uv setup follows [uv's integration guide](https://docs.astral.sh/uv/guides/integration/github/).

## Dependency updates

1. Update one related dependency group at a time in `pyproject.toml`, deliberately choosing versions. For mlx-audio, change the exact Git revision in `[tool.uv.sources]`; do not substitute the same-numbered PyPI package, which previously had different requirements. Assess Python patch updates separately from functional changes.
2. Run `uv lock`, review changed versions, sources, markers, and hashes, then run `bash scripts/setup.sh --dev`. Keep both metadata and lock in the change. Avoid freezing the active shared/global environment.
3. Run `bash scripts/check.sh`. Review backend probing, upstream signatures, catalog kwargs/capabilities, cache initialization, auxiliary repositories, and encoder behavior whenever MLX, mlx-audio, Transformers, HF Hub, NumPy, or audio libraries change.
4. On a Mac, run `bash run.sh --diag`, then generation using the affected families' actual catalog defaults. Test WAV and affected compressed formats, failure preservation, reference behavior, and offline reuse with internet connections blocked. Use disposable roots for first-run/repair tests. Detection/import success does not establish synthesis acceptance.
5. For execution/encoding changes, rerun the resource benchmark and inspect decoded PCM metadata/order. Keep [objective measurements](performance-2026-10-03.md) separate from listening. Obtain listening acceptance before declaring voice quality/consistency preserved.
6. Record versions, model commits, test commands, successes/failures, and unresolved gates in the checklist. Keep the known-good lock and model snapshots available for rollback; do not overwrite them merely to test a newer upstream snapshot.

## Model revisions and assets

Existing validated snapshots are reused. Normal online generation resolves and records an immutable revision for a new acquisition; it does not automatically replace a complete historical model with today's repository head. Primary and auxiliary completion have separate scopes. See [model storage](model-storage.md).

For a new model revision or family, first inspect its config/tokenizer/codec/voice requirements and pinned upstream loading code. Update the catalog's required assets and only the verified architecture adapters; unknown `model_type` values must fail rather than be relabeled broadly. Update auxiliary specs/revisions in `assets.py` when necessary, including file inventories. Test interruption/repair and offline missing-asset errors in disposable storage before replacing production assets. Preserve full precision and inspect real conversion behavior if conversion is required.

`RuntimeOptions.model_path_override` is an internal Python API, not a CLI flag. An existing local directory uses its local assets and explicit generation overrides; a `mlx-community/...` repo ID goes through the managed store, including pinning, staging, and offline checks. It is never passed remotely to the upstream loader. Local overrides require config and weights but do not establish managed inventory completeness; use normal catalog resolution for managed acquisition acceptance. `stream_wav=False` is an internal comparison/compatibility switch; the CLI always uses incremental WAV.

## Operator/model gates

The broad, expensive matrix is explicitly opt-in:

```bash
.venv/bin/python tests/integration_run_all_models.py --run --offline-second-pass
```

That command includes all configured families and may acquire large models, including Chatterbox and Whisper when transcription is needed. It has **not** been run for the current scope; do not use it blindly when those verifications are excluded. Supply nonempty matching reference transcripts to avoid unintended STT acquisition. Use a bounded per-family run instead, for example:

```bash
bash run.sh tests/fixtures/public_domain/alice_excerpt.txt \
  --model kokoro --offline --serial -o tests_output/kokoro-check.wav
```

The benchmark accepts `--model`, `--models-dir`, `--offline`, worker/batch limits, and timeouts for controlled per-family measurements. The current real-model acceptance scope is Kokoro and Soprano only. Their empty-root online acquisition/fresh-process offline reuse checks passed; evidence is in [the acquisition report](acquisition-2026-10-03.json). Larger-family acceptance is deferred and requires separately planned acquisition and sufficient disk/memory. Whisper/Chatterbox verification and release relocation/distribution are deferred/excluded for this delivery. Hosted CI, remaining-family generation, real conversion, and subjective listening remain visible acceptance gates in [the checklist](implementation-checklist.md).

## Current selected-model acceptance

On October 4, 2026, the user confirmed that both supplied default Kokoro/Soprano samples work as expected and preferred Kokoro's slightly more natural speech. Kokoro remains the default. This closes listening acceptance for those samples; it does not establish all-family or extended-duration speech acceptance. Larger-family tests and release portability remain deferred.

## Portable transfers

Technical Kokoro/Soprano relocation checks passed on October 4, 2026. Export with `.venv/bin/python scripts/export_portable_build.py`, move the source/model bundle, and recreate `.venv` at its destination using `bash scripts/setup.sh --no-dev`. Initial setup requires network access or a prepared dependency cache; included validated models then support offline generation. A personal bundle includes existing reference voices. [Portability evidence/instructions](portability-2026-10-04.md) describe export exclusions, guarded same-machine results, and the separate second-Mac/public-distribution gates. Earlier release-portability deferrals were reopened for this technical check; larger-family verification remains deferred.
