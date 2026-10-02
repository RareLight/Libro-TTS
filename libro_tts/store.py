from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shutil

import fcntl
from huggingface_hub import snapshot_download

from .catalog import (
    ModelSpec,
    canonical_model_type_for_key,
    is_mlx_repo_id,
)
from .paths import models_dir

SHARDED_SAFETENSOR_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.safetensors$"
)


@dataclass(frozen=True)
class ResolvedModel:
    model_key: str
    repo_id: str
    local_path: Path


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


class ModelStore:
    def __init__(
        self,
        root_dir: Path | None = None,
        snapshot_downloader=None,
        converter=None,
    ):
        self.root_dir = root_dir or models_dir()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = (root_dir or models_dir()) / "resolved_models.json"
        self._snapshot_downloader = snapshot_downloader or snapshot_download
        self._converter = converter
        self.logger = logging.getLogger("libro_tts.store")

    def ensure_model(self, spec: ModelSpec, offline: bool = False) -> Path:
        self._validate_mlx_repo_candidates(spec)

        lock_path = self.root_dir / f".{spec.key}.lock"
        with _file_lock(lock_path):
            existing = self._resolve_existing(spec)
            if existing is not None:
                self._ensure_model_type_metadata(spec, existing)
                return existing

            if offline:
                raise RuntimeError(
                    f"Model '{spec.key}' is not available locally and --offline was set."
                )

            errors: list[str] = []
            for repo_id in spec.repo_candidates:
                target = self._target_dir(spec.key, repo_id)
                try:
                    self._download_snapshot(repo_id=repo_id, target=target)
                    if self._has_ready_model(target):
                        self._ensure_model_type_metadata(spec, target)
                        self._update_manifest(spec.key, repo_id, target)
                        return target

                    self._convert_if_needed(repo_id=repo_id, target=target)
                    if self._has_ready_model(target):
                        self._ensure_model_type_metadata(spec, target)
                        self._update_manifest(spec.key, repo_id, target)
                        return target

                    errors.append(f"{repo_id}: downloaded but no ready MLX files")
                except Exception as exc:
                    errors.append(f"{repo_id}: {exc}")

            raise RuntimeError(
                f"Failed to resolve model '{spec.key}'. Attempted repos: "
                f"{', '.join(spec.repo_candidates)}. Errors: {' | '.join(errors)}"
            )

    def _resolve_existing(self, spec: ModelSpec) -> Path | None:
        manifest = self._load_manifest()
        model_entry = manifest.get("models", {}).get(spec.key)
        if model_entry:
            manifest_repo = str(model_entry.get("repo_id", ""))
            if manifest_repo not in spec.repo_candidates or not is_mlx_repo_id(manifest_repo):
                # Ignore stale/non-MLX manifest entries and continue with supported candidates.
                self.logger.info(
                    "Ignoring stale manifest entry for '%s': repo '%s' is not an allowed MLX candidate.",
                    spec.key,
                    manifest_repo,
                )
                model_entry = None

        if model_entry:
            manifest_path = Path(model_entry["path"])
            if self._has_ready_model(manifest_path):
                return manifest_path

        for repo_id in spec.repo_candidates:
            candidate = self._target_dir(spec.key, repo_id)
            if self._has_ready_model(candidate):
                self._update_manifest(spec.key, repo_id, candidate)
                return candidate

        return None

    def _download_snapshot(self, repo_id: str, target: Path) -> None:
        if target.exists() and any(target.iterdir()):
            if self._has_ready_model(target):
                return
            shutil.rmtree(target)

        target.mkdir(parents=True, exist_ok=True)
        self.logger.info("Downloading model '%s' into '%s'", repo_id, target)
        self._snapshot_downloader(repo_id=repo_id, local_dir=str(target))

    def _convert_if_needed(self, repo_id: str, target: Path) -> None:
        if self._has_ready_model(target):
            return

        converter = self._converter
        if converter is None:
            try:
                from mlx_audio.tts.utils import convert as convert_to_mlx
            except Exception:
                return
            converter = convert_to_mlx

        self.logger.info(
            "Attempting full-precision MLX conversion for '%s' into '%s'",
            repo_id,
            target,
        )
        converter(
            hf_path=repo_id,
            mlx_path=str(target),
            quantize=False,
            dequantize=False,
        )

    @staticmethod
    def _has_ready_model(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False

        config_path = path / "config.json"
        if not config_path.exists():
            return False

        weight_files = sorted(candidate for candidate in path.glob("*.safetensors") if candidate.is_file())
        if not weight_files:
            return False

        required_weight_files = ModelStore._required_weight_files_from_indexes(path)
        if required_weight_files is None:
            return False
        if required_weight_files:
            return all((path / filename).is_file() for filename in required_weight_files)

        shard_groups: dict[tuple[str, int], set[int]] = {}
        for weight_path in weight_files:
            match = SHARDED_SAFETENSOR_PATTERN.match(weight_path.name)
            if match is None:
                continue
            prefix = match.group("prefix")
            total = int(match.group("total"))
            shard_index = int(match.group("index"))
            shard_groups.setdefault((prefix, total), set()).add(shard_index)

        if shard_groups:
            return all(
                indices == set(range(1, total + 1))
                for (_, total), indices in shard_groups.items()
            )

        return True

    @staticmethod
    def _required_weight_files_from_indexes(path: Path) -> set[str] | None:
        index_paths = sorted(candidate for candidate in path.glob("*.index.json") if candidate.is_file())
        if not index_paths:
            return set()

        required_files: set[str] = set()
        for index_path in index_paths:
            try:
                with open(index_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                return None

            if not isinstance(payload, dict):
                return None

            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict):
                continue

            for value in weight_map.values():
                if isinstance(value, str) and value.endswith(".safetensors"):
                    required_files.add(value)

        return required_files

    @staticmethod
    def _safe_repo_id(repo_id: str) -> str:
        return repo_id.replace("/", "__")

    def _target_dir(self, model_key: str, repo_id: str) -> Path:
        return self.root_dir / model_key / self._safe_repo_id(repo_id)

    def _ensure_model_type_metadata(self, spec: ModelSpec, target: Path) -> None:
        config_path = target / "config.json"
        if not config_path.exists():
            return

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception:
            return

        if not isinstance(config, dict):
            return

        expected_model_type = canonical_model_type_for_key(spec.key)
        current_model_type = config.get("model_type")
        current_model_type_normalized = (
            str(current_model_type).strip().lower()
            if isinstance(current_model_type, str)
            else None
        )
        if current_model_type_normalized == expected_model_type:
            return

        config["model_type"] = expected_model_type
        try:
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, sort_keys=True)
            if current_model_type:
                self.logger.info(
                    "Normalized model_type from '%s' to '%s' in local model config at '%s'",
                    current_model_type,
                    expected_model_type,
                    config_path,
                )
            else:
                self.logger.info(
                    "Added missing model_type '%s' to local model config at '%s'",
                    expected_model_type,
                    config_path,
                )
        except Exception:
            return

    @staticmethod
    def _validate_mlx_repo_candidates(spec: ModelSpec) -> None:
        non_mlx = [repo for repo in spec.repo_candidates if not is_mlx_repo_id(repo)]
        if non_mlx:
            raise RuntimeError(
                f"Model '{spec.key}' has non-MLX candidates configured: {', '.join(non_mlx)}"
            )

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"version": 1, "models": {}}

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if not isinstance(data, dict):
                    return {"version": 1, "models": {}}
                data.setdefault("version", 1)
                data.setdefault("models", {})
                return data
        except Exception:
            return {"version": 1, "models": {}}

    def _update_manifest(self, model_key: str, repo_id: str, local_path: Path) -> None:
        data = self._load_manifest()
        data.setdefault("models", {})
        data["models"][model_key] = {
            "repo_id": repo_id,
            "path": str(local_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)


def resolve_model(
    model_store: ModelStore,
    spec: ModelSpec,
    offline: bool = False,
) -> ResolvedModel:
    local_path = model_store.ensure_model(spec=spec, offline=offline)
    manifest = model_store._load_manifest()
    repo_id = manifest.get("models", {}).get(spec.key, {}).get("repo_id", spec.repo_candidates[0])
    return ResolvedModel(model_key=spec.key, repo_id=repo_id, local_path=local_path)
