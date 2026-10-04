"""Auxiliary repositories required by the locked mlx-audio runtime.

Use the standard HF snapshot layout so upstream loaders with hard-coded repo
IDs can resolve these pinned assets without adapters or network access.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import RLock

from huggingface_hub import constants

from .store import COMMIT_PATTERN, COMPLETION_FILE, ModelStore, _file_lock


@dataclass(frozen=True)
class AssetSpec:
    key: str
    repo_id: str
    required_assets: tuple[str, ...]
    allow_patterns: tuple[str, ...] = ()


CSM_TOKENIZER = AssetSpec("csm_text", "unsloth/Llama-3.2-1B",
                          ("tokenizer.json", "tokenizer_config.json"), ("*.json",))
CSM_CODEC = AssetSpec("csm_mimi", "kyutai/moshiko-pytorch-bf16",
                     ("tokenizer-e351c8d8-checkpoint125.safetensors",),
                     ("tokenizer-e351c8d8-checkpoint125.safetensors",))
DIA_CODEC = AssetSpec("dia_dac", "mlx-community/descript-audio-codec-44khz",
                     ("config.json", "model.safetensors"), ("*.json", "*.safetensors"))
CHATTERBOX_TOKENIZER = AssetSpec("chatterbox_s3", "mlx-community/S3TokenizerV2",
                                ("config.json", "model.safetensors"), ("*.json", "*.safetensors"))
WHISPER = AssetSpec("whisper", "mlx-community/whisper-large-v3-turbo-asr-fp16",
                    ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "preprocessor_config.json"))


def _ready(store: ModelStore, path: Path, spec: AssetSpec) -> bool:
    for name in spec.required_assets:
        item = store._contained(path / name)
        if not item.is_file() or item.stat().st_size == 0:
            return False
        if name.endswith(".json"):
            try:
                if not isinstance(json.loads(item.read_text(encoding="utf-8")), dict):
                    return False
            except (OSError, ValueError):
                return False
    return True


def _atomic_ref(path: Path, revision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(revision)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_asset(store: ModelStore, spec: AssetSpec, offline: bool = False) -> Path:
    if (not re.fullmatch(r"[a-z0-9_]+", spec.key)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", spec.repo_id)):
        raise RuntimeError(f"Invalid auxiliary repository specification: {spec}")
    cache = store._contained(store.root_dir / ".hf/hub" / ("models--" + spec.repo_id.replace("/", "--")))
    ref = store._contained(cache / "refs/main")
    with _file_lock(store._contained(store.root_dir / f".asset-{store._safe_repo_id(spec.repo_id)}.lock")):
        entry = store._load_manifest().get("assets", {}).get(spec.key)
        revision = entry.get("revision") if entry and entry["repo_id"] == spec.repo_id else None
        if revision is None and ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
        if revision is not None and not COMMIT_PATTERN.fullmatch(revision):
            raise RuntimeError(f"Invalid cached revision for auxiliary '{spec.key}' at {ref}")
        if revision:
            target = store._contained(cache / "snapshots" / revision)
            completion = store._read_completion(target, spec.repo_id, scope="auxiliary_snapshot")
            if completion is not None and completion["revision"] != revision:
                completion = None
            if _ready(store, target, spec) and (completion or not (target / COMPLETION_FILE).exists()):
                if completion is None or not set(spec.required_assets).issubset(completion["required_assets"]):
                    recorded_spec = replace(spec, required_assets=tuple(sorted(set(spec.required_assets)
                                            | set(completion["required_assets"] if completion else ()))))
                    completion = store._record_completion(recorded_spec, spec.repo_id, target, revision,
                                                          completion["validation"] if completion else "local",
                                                          scope="auxiliary_snapshot")
                store._update_manifest(spec.key, spec.repo_id, target, completion, section="assets")
                _atomic_ref(ref, revision)
                return target
        if offline or constants.HF_HUB_OFFLINE:
            raise RuntimeError(f"Auxiliary asset '{spec.key}' ({spec.repo_id}) is missing or incomplete locally. "
                               f"Required: {', '.join(spec.required_assets)}. Run once online to acquire/repair it.")

        container = store._contained(store.root_dir / ".staging/assets" / spec.key / store._safe_repo_id(spec.repo_id))
        with asset_cache_environment(store.root_dir / ".hf"):
            plan = store._download_plan(spec.repo_id, container, spec.allow_patterns, revision=revision)
        target = store._contained(cache / "snapshots" / plan["revision"])
        staged = store._contained(container / "snapshot")
        # Keep existing cached assets available if acquisition is interrupted.
        if target.exists() and not staged.exists():
            for item in target.rglob("*"):
                store._contained(item)
            def copy_file(source, destination):
                try:
                    os.link(Path(source).resolve(), destination)
                except OSError:
                    shutil.copy2(source, destination)
            shutil.copytree(target, staged, copy_function=copy_file,
                            ignore=shutil.ignore_patterns(".cache", COMPLETION_FILE))
        staged.mkdir(parents=True, exist_ok=True)
        with asset_cache_environment(store.root_dir / ".hf"):
            store._snapshot_downloader(repo_id=spec.repo_id, revision=plan["revision"], local_dir=str(staged),
                                       **({"allow_patterns": list(spec.allow_patterns)} if spec.allow_patterns else {}))
        store._validate_download_files(staged, plan["files"])
        if not _ready(store, staged, spec):
            raise RuntimeError(f"Auxiliary '{spec.key}' lacks required assets: {', '.join(spec.required_assets)}")
        completion = store._record_completion(spec, spec.repo_id, staged, plan["revision"], "downloaded",
                                              scope="auxiliary_snapshot")
        store._publish_snapshot(staged, target, container)
        store._update_manifest(spec.key, spec.repo_id, target, completion, section="assets")
        _atomic_ref(ref, plan["revision"])
        return target


def prepare_model_assets(store: ModelStore, model_key: str, model_path: str, offline: bool) -> None:
    specs = {"csm": (CSM_TOKENIZER, CSM_CODEC), "dia": (DIA_CODEC,),
             "chatterbox": (CHATTERBOX_TOKENIZER,)}.get(model_key, ())
    if model_key == "csm":
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        tokenizer = config.get("text_tokenizer")
        if tokenizer:
            local = Path(tokenizer)
            if local.is_dir():
                if not local.resolve().is_relative_to(store.root_dir):
                    raise RuntimeError("CSM text_tokenizer path must be inside the active model store.")
                if not _ready(store, local, CSM_TOKENIZER):
                    raise RuntimeError(f"Incomplete CSM text tokenizer at {local}")
                specs = (CSM_CODEC,)
            else:
                specs = (AssetSpec("csm_text", tokenizer, CSM_TOKENIZER.required_assets,
                                   CSM_TOKENIZER.allow_patterns), CSM_CODEC)
    for spec in specs:
        ensure_asset(store, spec, offline)


_CACHE_LOCK = RLock()


def local_asset_loading(cache_root: Path | None = None):
    """Block HF acquisition during load/generate, including lazy lookups."""
    return asset_cache_environment(cache_root, local_only=True)


@contextmanager
def asset_cache_environment(cache_root: Path | None = None, *, local_only: bool = False):
    """Select the active store for already-imported HF loaders and children.

    HF 1.8 reads its constants dynamically; updating both constants and env is
    necessary for already-imported loaders and child processes. Local-only
    loading follows managed acquisition. The CLI does not synthesize in threads.
    """
    with _CACHE_LOCK:
        root = cache_root.resolve() if cache_root is not None else Path(constants.HF_HOME)
        values = {"HF_HOME": str(root), "HF_HUB_CACHE": str(root / "hub"),
                  "HUGGINGFACE_HUB_CACHE": str(root / "hub"),
                  "HF_ASSETS_CACHE": str(root / "assets"), "HF_XET_CACHE": str(root / "xet"),
                  "HUGGINGFACE_ASSETS_CACHE": str(root / "assets"),
                  "TRANSFORMERS_CACHE": str(root / "transformers"), "HF_DATASETS_CACHE": str(root / "datasets"),
                  "HF_TOKEN_PATH": str(root / "token"), "HF_STORED_TOKENS_PATH": str(root / "stored_tokens")}
        if local_only:
            values.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        env_before = {name: os.environ.get(name) for name in values}
        constants_before = {name: getattr(constants, name) for name in values if hasattr(constants, name)}
        try:
            os.environ.update(values)
            for name in constants_before:
                setattr(constants, name, True if name == "HF_HUB_OFFLINE" else values[name])
            yield
        finally:
            for name, value in constants_before.items():
                setattr(constants, name, value)
            for name, value in env_before.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
