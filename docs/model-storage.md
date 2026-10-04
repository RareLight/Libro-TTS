# Model storage and asset coverage

Implemented October 3, 2026 against the locked mlx-audio revision `e42e1431fcf89af313375296c46d03a0153c4aa7`. This document covers primary and auxiliary storage; implementation and real model-family acceptance are tracked separately.

## Acquisition and recovery

The published location remains `models/<model_key>/<repo_id_sanitized>/`. Existing compatible weights are reused; generation does not automatically upgrade a validated model.

For a new or incomplete primary snapshot, the store reads repository metadata, records a full commit SHA and file-size inventory in `models/.staging/<key>/<repo>/download.json`, and downloads that revision into the sibling `snapshot/` directory. An interrupted retry uses the same plan and HF local-dir metadata. Incomplete historical directories are moved into staging, preserving weights and resumable files rather than deleting them.

Before publication, every file in the repository inventory must exist with its expected size where available. Config must be a JSON object; weights must be nonempty, shard groups and weight indexes complete, and the catalog's required primary assets present. Bundled Qwen and Spark codec weights receive nested shard/index checks. Only then are metadata normalized and a completion inventory written. Renaming the validated directory publishes it at the usual location.

If a fully downloaded source needs MLX conversion, input and output use separate local staging directories. Conversion retains `quantize=False` and `dequantize=False`. Source files remain available for retry; the converted result must pass the same runtime asset checks. Conversion with actual model weights has not been exercised in this delivery.

The model lock serializes acquisition of one family. A separate short shared lock covers manifest read/update/replace across different families. JSON writes flush to a temporary sibling and atomically replace the destination. Existing output audio is unaffected by these storage changes; atomic audio output remains a separate implementation stage.

Do not remove staging while a download is running or if its resumable progress is needed. On rare conflicting repair state, prior files are retained as `previous-*` within that transaction's staging directory. Staging is excluded from portable exports. Malformed manifest or download-plan JSON produces an actionable error and is preserved for repair.

## Completion metadata and migration

`resolved_models.json` uses version 2 and paths relative to the active store. It records repository ID, revision, required assets, completion scope, validation source, and update time. `.libro-model.json` inside each published snapshot stores its validated file-size inventory.

`complete: true` is scoped to `primary_snapshot` for `models` entries and `auxiliary_snapshot` for `assets` entries. A primary completion record does not assert that its auxiliary repositories are available. Neither scope means that synthesis succeeded for all models. Normal reuse checks file presence and sizes; it does not hash gigabytes of weights or detect every possible same-size corruption. Required auxiliary JSON files must also parse as objects.

Historical absolute entries never select files in another checkout. The store validates the current root's canonical candidate and migrates that entry in place. Paths and symlinks that escape the active store are rejected. Structurally valid historical snapshots are recorded as `validation: local`; new downloaded and converted snapshots use `downloaded` and `converted`. Historical revision is recorded only when HF metadata for all required primary files has one consistent SHA. Missing or mixed provenance yields `revision: null`; today's remote revision is never substituted for old files.

A malformed completion marker or missing/resized inventory file makes the snapshot unavailable offline and eligible for online repair. Invalid manifest JSON/schema stops acquisition before downloads or overwrite. Migration happens on successful model resolution; unused stale entries are preserved until their model is resolved.

## Primary assets checked

All families require valid config and complete nonempty weights. These additional requirements come from the inspected runtime loaders and repository inventories:

| Family | Required assets in the primary snapshot | Separate managed dependencies / remaining coverage |
|---|---|---|
| Kokoro | Included default `voices/af_aoede.safetensors`; chosen named/blended voices resolve locally during generation | None for the verified English default; other language/package paths require separate coverage |
| Qwen3-TTS | `tokenizer_config.json`, `vocab.json`, `merges.txt`, speech-tokenizer config and weights | No additional dependency identified for Libro's default named-voice path |
| CSM | No extra files in the primary repo required by this check | Text tokenizer selected by config or `unsloth/Llama-3.2-1B`; Mimi tensor in `kyutai/moshiko-pytorch-bf16`; optional upstream prompt lookup |
| Dia | No extra files in the primary repo required by this check | `mlx-community/descript-audio-codec-44khz` |
| Spark | Text tokenizer/config, audio-tokenizer YAML, BiCodec YAML/weights, wav2vec config/weights/preprocessor | Reference transcription when no text is supplied |
| Chatterbox | `tokenizer.json` | Managed `mlx-community/S3TokenizerV2`; alternate multilingual runtime/tokenizer assets require additional coverage |
| Soprano | `tokenizer.json`, `tokenizer_config.json` | None identified for the verified default path |
| Voxtral | `tekken.json`, included default `voice_embedding/neutral_female.safetensors` | Every requested voice must have a nonempty contained local embedding; real voice synthesis remains an acceptance gate |

## Auxiliary acquisition and runtime loading

`assets.py` declares the separate repositories used by the locked runtime. Auxiliary files use HF's standard `models/.hf/hub/models--<namespace>--<repo>/snapshots/<commit>/` layout with an atomically written `refs/main`. This allows upstream CSM/Dia/Chatterbox loaders with hard-coded repository IDs to find the pinned files without patching those loaders. The layout follows the official [cache documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) and is tested against the locked hub 1.8.0 implementation.

Acquisition uses per-repository locks, filtered file-size inventories where appropriate, resumable staging under `.staging/assets/<key>/<repo>/`, validated publication, and the existing shared manifest transaction. CSM downloads tokenizer JSON only from its configured text repository (default `unsloth/Llama-3.2-1B`) and only the Mimi tensor from `kyutai/moshiko-pytorch-bf16`; Llama and Moshi language-model weights are excluded. Dia and Chatterbox acquire their codec config/weights. Existing cache snapshots, including relative blob symlinks, are reused if all required files and any completion inventory validate inside the active root. Historical provenance comes from the cached commit, with `validation: local`, rather than current remote metadata. Interrupted repairs retain the previous cache snapshot.

Reference transcription acquires the full `mlx-community/whisper-large-v3-turbo-asr-fp16` snapshot only when needed. It uses the supported `mlx_audio.stt.load(<local_path>)` API; the deprecated Whisper class method and implicit repository download are removed. STT runtime reuse is keyed by local path. The model weights alone are about 1.6 GB. Missing or incomplete offline assets fail before model loading.

Named/blended Kokoro voices resolve to nonempty included tensors or managed acquisition of only the selected tensor from `prince-canuma/Kokoro-82M`. Explicit tensor paths remain supported. Voxtral selected embeddings are validated before TTS loading so an absent voice cannot silently fall back to unconditioned generation.

Acquisition selects the active store's HF home/hub/Xet/assets/credentials paths. Runtime loading and lazy generation then run under `local_asset_loading`, which sets both environment and already-imported hub constants to local-only policy. The state is restored on normal/exceptional exit. Spawned workers receive the active cache root explicitly. This uses hub 1.8.0's dynamic offline/cache constants; dependency upgrades must rerun these tests. The CLI uses one synthesis thread per process; this context is not a general guarantee for unrelated concurrent threaded Hub clients.

The locked English/default paths are covered. Optional CSM watermarking (`silentcipher`, absent from the locked environment), alternate multilingual Chatterbox assets/segmentation, and additional Kokoro language package paths have not been added or validated. New model/runtime variants require explicit asset and dependency coverage before being considered portable.

## Revision and validation evidence

Repository metadata was read October 3, 2026 without downloading weights. These are observed revisions, not new globally hardcoded model pins or all-family synthesis acceptance:

| Repository | Observed commit |
|---|---|
| `mlx-community/Kokoro-82M-bf16` | `a71e4d38b236d968966a2002c4c895dbd12b1c3c` |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | `a6eb4f68e4b056f1215157bb696209bc82a6db48` |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | `1eccf1cb2519b5a4e8a95b5f0544f3303568164f` |
| `mlx-community/csm-1b` | `5bf5ec118cf45fecc7b51198fd9f1a20a5aab65a` |
| `mlx-community/Dia-1.6B-fp16` | `4575dd9622ffa7dd14cc4342e1aac4bb7841904d` |
| `mlx-community/Spark-TTS-0.5B-bf16` | `cac753e3cf9d9d92524eb15fe256efc7db45ce7e` |
| `mlx-community/chatterbox-fp16` | `4923fcca09086356aeab5191a2348c5a17a23694` |
| `mlx-community/Soprano-1.1-80M-bf16` | `745350c27f356c3910eebcb49e29760dcf6a643c` |
| `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` | `dd85c02adbae551f5bb29ded35ee60ccdfb90927` |

The metadata API and full revision selection follow the official [HfApi documentation](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.model_info) and [download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

The fast suite passed 122 tests, including 22 new storage regressions. Real Kokoro and Soprano synthesis used default catalog resolution against disposable copied trees, migrated absolute manifests, an empty HF cache, blocked network connections, and one synthesis worker. WAV metadata was verified: Kokoro 24,000 Hz/79,200 frames; Soprano 32,000 Hz/83,968 frames; both mono. Their consistent local revision metadata matched the observed commits above. The original model tree and manifest were not migrated by these fixtures.

Real first-run primary acquisition, the remaining six families, live conversion, relocated full release bundles, and human listening remain unverified. Temporary evidence: `/private/tmp/libro-store-repo-inventory.json`, `/private/tmp/libro-store-full-tests.log`, `/private/tmp/libro-store-real-smoke.log`, and `/private/tmp/libro-store-real-smoke/`.

### Auxiliary delivery evidence

- Fast suite: 142 tests passed. New cases cover all declared auxiliary sets, filtered/pinned acquisition, interrupted retry/repair, malformed metadata, truncated files, relocated blob symlinks, external-path rejection, real HF cache lookup with network blocked, context restoration, custom CSM text repositories, local STT API/path reuse, selected voice checks, and fresh spawned worker cache lookup.
- Real offline checks used copied project-cache data: CSM's tokenizer loaded and encoded text; Mimi and Dia DAC loaded and evaluated their parameters with network connections blocked. Known cached commits were `9535bd9b1d1dea6acafbdc4813b728796aeb28da`, `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`, and `91dff27c81edbb0992be17ac51d2c7d515411a32`, respectively. This is auxiliary-load evidence, not CSM/Dia TTS acceptance.
- Real selective first-run acquisition downloaded five tokenizer JSON files and one Kokoro voice tensor to a disposable root, then reused both offline. No Llama model weights were acquired. Revisions were `9535bd9b1d1dea6acafbdc4813b728796aeb28da` and `e02c9eada7ce7416798af36b190a8a2dd2ecd566`.
- Kokoro and Soprano synthesized under local-only loading from copied primary trees: 24,000 Hz/79,200 frames and 32,000 Hz/90,112 frames, respectively; mono. Original model metadata was not migrated by these fixtures.
- Whisper/S3Tokenizer file inventories were checked against remote metadata; their approximately 1.6 GB/495 MB weights were not downloaded. Full STT, Chatterbox synthesis, remaining-family TTS, complete release relocation, and listening remain open.

Temporary evidence: `/private/tmp/libro-assets-full-tests.log`, `/private/tmp/libro-assets-real-smoke.log`, `/private/tmp/libro-assets-online-smoke.log`, and their corresponding fixture directories.
