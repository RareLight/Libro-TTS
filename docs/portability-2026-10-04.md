# Portability verification — October 4, 2026

Kokoro and Soprano passed relocated-bundle setup and offline generation on this Apple Silicon Mac. The transfer format is a source/model folder with a locked installation recipe. It is not a copied virtual environment or an installer containing Python and all dependency wheels. A second physical Mac remains an operator acceptance gate.

## Export fixes

`export_portable_build.py` now:

- Validates included symlinks before copying, rejecting external, broken/cyclic, or excluded-metadata targets. Internal blob/file links are preserved as relative links, including formerly absolute internal links.
- Rejects output inside a copied source tree, preventing recursive export. Existing destinations and dangling destination symlinks are preserved/rejected.
- Builds in an owned temporary sibling and publishes only a complete bundle. Copy/validation failures clean up staging instead of leaving a misleading final bundle.
- Adopts included primary snapshots using local-only resolution in the copied store. Known revisions are captured before removing download metadata. Manifest version 2 contains relative paths; unavailable historical entries are omitted. Auxiliary manifest paths are relocated or pruned when unavailable.
- Excludes HF credential filenames, locks, incomplete files, temporary files/logs, staging, download metadata, and Xet chunk/log cache. Models, necessary hub blobs/snapshots, private reference voices, app license, existing model cards/notices, lock/setup contract, and source-reference docs remain included.
- Allows a source-only export when optional model/reference folders are absent. Such a bundle needs online model acquisition before offline generation.

The original manifest and both source model configs were compared against the baseline export and remained byte-identical. Source weights/caches were not migrated or repaired by these checks. Existing defaults and dependency pins were preserved.

## Live verification

A baseline export installed the lock using a genuinely fresh dedicated uv cache and managed Python install. The fixed export was moved into a path containing spaces **before** setup, then installed offline from the cache populated by that first install. It was subsequently moved again **after** setup; only that disposable `.venv` was removed, and a new environment was installed at the second location.

Both recreated runtime environments used Python **3.12.13**, the same pinned application dependencies, and **114** local distributions (`--no-dev`). Package origins were all inside each destination `.venv`; system/user site packages were excluded. The managed base interpreter was outside the original checkout. FFmpeg 7.1 resolved inside the destination wheel through a relative `.venv/bin/ffmpeg` symlink. Dependency consistency passed at the second location. The existing source environment remains unchanged on Python 3.12.2.

Generation ran from an unrelated caller directory with spaces in the input/bundle paths. Deliberately conflicting inherited Python/HF paths were supplied; the launcher cleared Python overrides and bootstrap selected the new local cache. Homebrew was absent from runtime `PATH`. Temporary destination-only `sitecustomize` hooks rejected Python filesystem access to the original checkout/global HF/Torch/uv caches and internet socket connections; negative probes confirmed the hooks were active. Hooks were inherited by model workers and isolated preflight subprocesses, then removed in `finally`. This is a guarded same-machine test, not an OS-level assertion about every possible native library access.

Nine model/cache symlinks resolved inside the relocated model tree, with no absolute/dangling links. Exported primary manifest entries were relative and complete for Kokoro/Soprano; six unavailable legacy family entries were removed. No other TTS family was acquired or generated.

| First relocated run | Mode | Decoded frames | Rate / channels |
|---|---|---:|---|
| Kokoro WAV | serial | 779,400 | 24,000 Hz / mono |
| Kokoro WAV | two workers, batch two | 779,400 | 24,000 Hz / mono |
| Soprano WAV | serial | 929,792 | 32,000 Hz / mono |
| Soprano WAV | two workers, batch two | 954,368 | 32,000 Hz / mono |
| Kokoro MP3 | serial | 105,000 | 24,000 Hz / mono |
| Kokoro FLAC | serial | 105,000 | 24,000 Hz / mono |

Diagnostics and model listing passed. Parallel runs used the requested mode without serial fallback. All decoded waveforms were finite and nonzero; previous WAV placeholders were replaced and no owned audio staging files remained. Soprano duration varies stochastically. After the second move/environment recreation, diagnostics plus Kokoro WAV/MP3 passed again with the same guard conditions and frame counts.

The full fast maintenance gate passed **245 tests**, dependency consistency, Ruff, Bash syntax, and blocked-network commands/imports. Seven exporter tests cover exclusions, optional assets, source preservation, primary/auxiliary migration, relative blob links surviving original-tree removal, external/broken/cyclic/credential/Xet links, recursive/existing destinations, and incomplete-export cleanup. [Retained machine-readable evidence](portability-2026-10-04.json) contains both live reports; temporary logs/artifacts are under `/private/tmp/libro-portability-*`.

## Transfer and setup

On October 4, 2026, the user copied the prepared package to a new location, tested it, and reported that everything worked. This closes the user's location-transfer acceptance. The report did not specify a different physical Mac, so the separate cross-machine boundary below remains unverified.

From the source checkout:

```bash
.venv/bin/python scripts/export_portable_build.py --output-dir dist/Libro-TTS-portable
```

Copy/move the resulting folder to another Apple Silicon Mac, install uv if needed, and run there:

```bash
bash scripts/setup.sh --no-dev
bash run.sh --diag
bash run.sh your_text.txt -o output.wav --offline
```

Initial Python/package installation requires internet access or a separately prepared dependency cache. Downloaded model snapshots are already in the bundle; verified Kokoro/Soprano generation then works offline. If moving an installed bundle, recreate its `.venv` at the new path. Preserve `models/` and `reference_voices/` when doing so. The bundle's docs describe source-checkout development gates; the minimal runtime export excludes the test suite.

## Remaining boundaries

- Actual execution on a second physical Apple Silicon Mac is unverified; Windows/Linux/Intel support is outside the MLX runtime contract.
- Larger-family speech/auxiliary/conversion acceptance remains deferred. Exported cached data does not establish those families' speech readiness.
- The app license and existing model cards/notices are retained. This technical transfer check does not close a public-distribution review of dependency/model licenses and any required source/notices. In particular, the copied Soprano conversion card refers readers to its upstream model card.
- Private reference audio/text is included when present. This is a personal transfer artifact, not a published release. No new subjective listening test was performed; the user previously accepted the supplied Kokoro/Soprano samples.
