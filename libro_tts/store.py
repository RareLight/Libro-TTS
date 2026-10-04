from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4
from typing import TYPE_CHECKING

import fcntl
from fnmatch import fnmatch
from huggingface_hub import HfApi, constants, snapshot_download

from .catalog import (
    ModelSpec,
    canonical_model_type_for_key,
    MODEL_CONFIG_TYPE_ADAPTERS,
)
from .paths import models_dir

if TYPE_CHECKING:
    from .assets import AssetSpec

SHARDED_SAFETENSOR_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.safetensors$"
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
COMPLETION_FILE = ".libro-model.json"


def _safe_relative_name(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\\" not in value


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Keep readers and interrupted writers from observing truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        repo_info_loader=None,
    ):
        self.root_dir = (root_dir or models_dir()).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root_dir / "resolved_models.json"
        self._snapshot_downloader = snapshot_downloader or snapshot_download
        self._converter = converter
        self._repo_info_loader = repo_info_loader or HfApi().model_info
        self.logger = logging.getLogger("libro_tts.store")

    def ensure_model(self, spec: ModelSpec, offline: bool = False) -> Path:
        self._validate_mlx_repo_candidates(spec)
        offline = offline or constants.HF_HUB_OFFLINE

        lock_path = self._contained(self.root_dir / f".{spec.key}.lock")
        with _file_lock(lock_path):
            existing = self._resolve_existing(spec)
            if existing is not None:
                return existing

            if offline:
                raise RuntimeError(
                    f"Model '{spec.key}' has no complete local primary snapshot and offline mode is active. "
                    "Run once online to acquire/repair its weights and required tokenizer/voice assets."
                )

            errors: list[str] = []
            for repo_id in spec.repo_candidates:
                target = self._target_dir(spec.key, repo_id)
                try:
                    container = self._contained(self.root_dir / ".staging" / spec.key / self._safe_repo_id(repo_id))
                    plan = self._download_plan(repo_id, container)
                    staged = self._contained(container / "snapshot")
                    if target.exists() and not staged.exists():
                        # Retain weights and HF's resumable local-dir metadata.
                        staged.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(target, staged)
                    self._download_snapshot(repo_id, staged, plan["revision"])
                    self._validate_download_files(staged, plan["files"])
                    validation = "downloaded"
                    if not self._weights_ready(staged):
                        staged = self._convert_if_needed(repo_id, staged)
                        validation = "converted"
                    if not self._has_ready_model(staged, spec):
                        raise RuntimeError("Snapshot is missing valid config, complete weights, or required assets: "
                                           + ", ".join(spec.required_assets))
                    self._ensure_model_type_metadata(spec, staged)
                    completion = self._record_completion(spec, repo_id, staged, plan["revision"], validation)
                    self._publish_snapshot(staged, target, container)
                    self._update_manifest(spec.key, repo_id, target, completion)
                    return target
                except Exception as exc:
                    errors.append(f"{repo_id}: {exc}")

            raise RuntimeError(
                f"Failed to resolve model '{spec.key}'. Attempted repos: "
                f"{', '.join(spec.repo_candidates)}. Errors: {' | '.join(errors)}"
            )

    def _resolve_existing(self, spec: ModelSpec) -> Path | None:
        manifest = self._load_manifest()
        model_entry = manifest.get("models", {}).get(spec.key)
        candidates = list(spec.repo_candidates)
        if model_entry and model_entry["repo_id"] in candidates:
            repo_id = model_entry["repo_id"]
            stored = Path(model_entry["path"])
            expected = self._target_dir(spec.key, repo_id)
            # Absolute v1 entries are migrated by validating the active root's
            # canonical candidate, never by following the old checkout path.
            if stored.is_absolute() or (
                _safe_relative_name(str(stored)) and self.root_dir / stored == expected
            ):
                candidates.remove(repo_id)
                candidates.insert(0, repo_id)
            else:
                self.logger.warning("Ignoring noncanonical manifest path for '%s': %s", spec.key, stored)

        for repo_id in candidates:
            candidate = self._target_dir(spec.key, repo_id)
            if not self._has_ready_model(candidate, spec):
                continue
            completion = self._read_completion(candidate, repo_id)
            if (candidate / COMPLETION_FILE).exists() and completion is None:
                continue
            changed = self._ensure_model_type_metadata(spec, candidate)
            if completion is None or changed:
                completion = self._record_completion(
                    spec, repo_id, candidate,
                    completion["revision"] if completion else self._local_revision(candidate, spec),
                    completion["validation"] if completion else "local",
                )
            self._update_manifest(spec.key, repo_id, candidate, completion)
            return candidate
        return None

    def _download_snapshot(self, repo_id: str, target: Path, revision: str) -> None:
        target.mkdir(parents=True, exist_ok=True)
        self.logger.info("Downloading model '%s' into '%s'", repo_id, target)
        self._snapshot_downloader(repo_id=repo_id, local_dir=str(target), revision=revision)

    def _convert_if_needed(self, repo_id: str, target: Path) -> Path:
        if self._weights_ready(target):
            return target

        converter = self._converter
        if converter is None:
            try:
                from mlx_audio.tts.utils import convert as convert_to_mlx
            except Exception:
                return target
            converter = convert_to_mlx

        converted = self._contained(target.parent / "converted")
        converted.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Attempting full-precision MLX conversion for '%s' into '%s'",
            repo_id,
            converted,
        )
        converter(
            hf_path=str(target),
            mlx_path=str(converted),
            quantize=False,
            dequantize=False,
        )
        return converted

    @staticmethod
    def _has_ready_model(path: Path, spec: ModelSpec | None = None) -> bool:
        if not path.exists() or not path.is_dir():
            return False

        config_path = path / "config.json"
        if not ModelStore._local_file(path, config_path):
            return False
        try:
            if not isinstance(json.loads(config_path.read_text(encoding="utf-8")), dict):
                return False
        except (OSError, ValueError):
            return False
        if not ModelStore._weights_ready(path):
            return False
        for pattern in spec.required_assets if spec else ():
            matches = list(path.glob(pattern))
            if not matches or not all(ModelStore._local_file(path, item) for item in matches):
                return False
            if pattern.endswith("*.safetensors") and not ModelStore._weights_ready(matches[0].parent):
                return False
        return True

    @staticmethod
    def _local_file(root: Path, path: Path) -> bool:
        return path.is_file() and path.resolve().is_relative_to(root.resolve()) and path.stat().st_size > 0

    @staticmethod
    def _weights_ready(path: Path) -> bool:
        weight_files = sorted(candidate for candidate in path.glob("*.safetensors") if candidate.is_file())
        if not weight_files or not all(ModelStore._local_file(path, item) for item in weight_files):
            return False

        required_weight_files = ModelStore._required_weight_files_from_indexes(path)
        if required_weight_files is None:
            return False
        if required_weight_files:
            return all(ModelStore._local_file(path, path / filename) for filename in required_weight_files)

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
            if not ModelStore._local_file(path, index_path):
                return None
            try:
                with open(index_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                return None

            if not isinstance(payload, dict):
                return None

            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                return None

            for value in weight_map.values():
                if not isinstance(value, str) or not _safe_relative_name(value) or not value.endswith(".safetensors"):
                    return None
                required_files.add(value)

        return required_files

    @staticmethod
    def _safe_repo_id(repo_id: str) -> str:
        return repo_id.replace("/", "__")

    def _target_dir(self, model_key: str, repo_id: str) -> Path:
        return self._contained(self.root_dir / model_key / self._safe_repo_id(repo_id))

    def _contained(self, path: Path) -> Path:
        if not path.resolve().is_relative_to(self.root_dir):
            raise RuntimeError(f"Model-store path escapes the active root: {path}")
        return path

    def _download_plan(self, repo_id: str, container: Path, allow_patterns: tuple[str, ...] = (),
                       *, revision: str | None = None) -> dict:
        path = self._contained(container / "download.json")
        if path.exists():
            try:
                plan = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Cannot read resumable download metadata at {path}: {exc}") from exc
        else:
            info = self._repo_info_loader(repo_id=repo_id, files_metadata=True,
                                          **({"revision": revision} if revision else {}))
            plan = {"repo_id": repo_id, "revision": info.sha,
                    "files": {item.rfilename: item.size for item in info.siblings
                              if not allow_patterns or any(fnmatch(item.rfilename, pattern) for pattern in allow_patterns)}}
            if allow_patterns:
                plan["allow_patterns"] = list(allow_patterns)
        if (
            not isinstance(plan, dict) or plan.get("repo_id") != repo_id
            or not isinstance(plan.get("revision"), str) or not COMMIT_PATTERN.fullmatch(plan["revision"])
            or not self._valid_inventory(plan.get("files"), allow_unknown_size=True)
            or COMPLETION_FILE in plan["files"]
            or plan.get("allow_patterns", []) != list(allow_patterns)
            or revision is not None and plan.get("revision") != revision
        ):
            raise RuntimeError(f"Invalid resumable download metadata for '{repo_id}' at {path}")
        if not path.exists():
            _atomic_write_json(path, plan)
        return plan

    @staticmethod
    def _valid_inventory(files, *, allow_unknown_size: bool = False) -> bool:
        return isinstance(files, dict) and bool(files) and all(
            isinstance(name, str) and _safe_relative_name(name)
            and (type(size) is int and size >= 0 or allow_unknown_size and size is None)
            for name, size in files.items()
        )

    def _validate_download_files(self, path: Path, files: dict) -> None:
        for name, size in files.items():
            item = self._contained(path / name)
            if not item.is_file() or size is not None and item.stat().st_size != size:
                raise RuntimeError(f"Incomplete snapshot: missing or truncated '{name}' in {path}")

    def _record_completion(
        self, spec: ModelSpec | AssetSpec, repo_id: str, path: Path, revision: str | None, validation: str,
        *, scope: str = "primary_snapshot",
    ) -> dict:
        files = {}
        for item in path.rglob("*"):
            self._contained(item)
            relative = item.relative_to(path)
            if item.is_file() and ".cache" not in relative.parts and item.name not in {COMPLETION_FILE, ".DS_Store"}:
                files[relative.as_posix()] = item.stat().st_size
        completion = {
            "version": 1, "scope": scope, "repo_id": repo_id,
            "revision": revision, "validation": validation,
            "required_assets": list(spec.required_assets), "files": files,
        }
        try:
            _atomic_write_json(self._contained(path / COMPLETION_FILE), completion)
        except OSError as exc:
            raise RuntimeError(f"Cannot record snapshot completion at {path}: {exc}") from exc
        return completion

    def _read_completion(self, path: Path, repo_id: str, *, scope: str = "primary_snapshot") -> dict | None:
        marker = self._contained(path / COMPLETION_FILE)
        if not marker.exists():
            return None
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            if (
                not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1
                or data.get("scope") != scope or data.get("repo_id") != repo_id
                or data.get("validation") not in {"local", "downloaded", "converted"}
                or not self._valid_inventory(data.get("files"))
                or not isinstance(data.get("required_assets"), list)
                or not all(isinstance(name, str) and _safe_relative_name(name) for name in data["required_assets"])
                or (data.get("revision") is not None and (
                    not isinstance(data["revision"], str) or not COMMIT_PATTERN.fullmatch(data["revision"])
                ))
            ):
                return None
            self._validate_download_files(path, data["files"])
            return data
        except (OSError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _local_revision(path: Path, spec: ModelSpec) -> str | None:
        # HF local_dir metadata stores the commit on its first line. Missing or
        # mixed metadata is explicitly unknown; do not guess today's revision.
        assets = [path / "config.json", *path.glob("*.safetensors")]
        for pattern in spec.required_assets:
            assets.extend(path.glob(pattern))
        revisions = set()
        for asset in assets:
            relative = asset.relative_to(path)
            item = path / ".cache/huggingface/download" / f"{relative.as_posix()}.metadata"
            if not item.resolve().is_relative_to(path.resolve()):
                return None
            try:
                first = item.read_text(encoding="utf-8").splitlines()
            except (OSError, ValueError):
                return None
            if not first or not COMMIT_PATTERN.fullmatch(first[0]):
                return None
            revisions.add(first[0])
        return next(iter(revisions)) if len(revisions) == 1 else None

    def _publish_snapshot(self, staged: Path, target: Path, container: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = None
        if target.exists():
            previous = self._contained(container / f"previous-{uuid4().hex}")
            os.replace(target, previous)
        try:
            os.replace(staged, target)
        except OSError:
            if previous is not None:
                os.replace(previous, target)
            raise

    def _ensure_model_type_metadata(self, spec: ModelSpec, target: Path) -> bool:
        config_path = target / "config.json"
        if not config_path.exists():
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception:
            return False

        if not isinstance(config, dict):
            return False

        expected_model_type = canonical_model_type_for_key(spec.key)
        current_model_type = config.get("model_type")
        current_model_type_normalized = (
            str(current_model_type).strip().lower()
            if isinstance(current_model_type, str)
            else None
        )
        if current_model_type == expected_model_type:
            return False

        if current_model_type_normalized and current_model_type_normalized != expected_model_type and current_model_type_normalized not in MODEL_CONFIG_TYPE_ADAPTERS.get(spec.key, ()):
            raise RuntimeError(f"Unexpected model_type {current_model_type!r} for '{spec.key}' at {config_path}; "
                               "refusing to relabel this architecture.")
        if current_model_type is not None and not isinstance(current_model_type, str):
            raise RuntimeError(f"Invalid non-string model_type for '{spec.key}' at {config_path}.")

        config["model_type"] = expected_model_type
        try:
            _atomic_write_json(self._contained(config_path), config)
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
        except OSError as exc:
            raise RuntimeError(f"Cannot normalize model metadata at {config_path}: {exc}") from exc
        return True

    @staticmethod
    def _validate_mlx_repo_candidates(spec: ModelSpec) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", spec.key):
            raise RuntimeError(f"Invalid model key: {spec.key}")
        if not spec.repo_candidates or any(not re.fullmatch(r"mlx-community/[A-Za-z0-9_.-]+", repo) for repo in spec.repo_candidates):
            raise RuntimeError(f"Invalid MLX repository candidates for '{spec.key}'")

    def _load_manifest(self) -> dict:
        self._contained(self.manifest_path)
        if not self.manifest_path.exists():
            return {"version": 2, "models": {}}

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or type(data.get("version", 1)) is not int or data.get("version", 1) not in {1, 2}:
                raise ValueError("unsupported manifest schema/version")
            if not isinstance(data.get("models"), dict):
                raise ValueError("'models' must be an object")
            if not isinstance(data.get("assets", {}), dict):
                raise ValueError("'assets' must be an object")
            for key, entry in list(data["models"].items()) + list(data.get("assets", {}).items()):
                if not isinstance(entry, dict) or not all(isinstance(entry.get(name), str) for name in ("repo_id", "path")):
                    raise ValueError(f"invalid model entry '{key}'")
                if "complete" in entry and type(entry["complete"]) is not bool:
                    raise ValueError(f"invalid completion flag for '{key}'")
                revision = entry.get("revision")
                if revision is not None and (not isinstance(revision, str) or not COMMIT_PATTERN.fullmatch(revision)):
                    raise ValueError(f"invalid revision for '{key}'")
                if "required_assets" in entry and (
                    not isinstance(entry["required_assets"], list)
                    or not all(isinstance(name, str) and _safe_relative_name(name) for name in entry["required_assets"])
                ):
                    raise ValueError(f"invalid required assets for '{key}'")
            return data
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read model manifest at {self.manifest_path}: {exc}. "
                               "Preserve the file and repair its metadata before retrying.") from exc

    def _update_manifest(self, model_key: str, repo_id: str, local_path: Path, completion: dict,
                         *, section: str = "models") -> None:
        relative = self._contained(local_path).relative_to(self.root_dir).as_posix()
        with _file_lock(self._contained(self.root_dir / ".manifest.lock")):
            data = self._load_manifest()
            data["version"] = 2
            data.setdefault(section, {})[model_key] = {
                "repo_id": repo_id, "path": relative, "complete": True,
                "scope": completion["scope"], "revision": completion["revision"],
                "validation": completion["validation"],
                "required_assets": completion["required_assets"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                _atomic_write_json(self.manifest_path, data)
            except OSError as exc:
                raise RuntimeError(f"Cannot update model manifest at {self.manifest_path}: {exc}") from exc


def resolve_model(
    model_store: ModelStore,
    spec: ModelSpec,
    offline: bool = False,
) -> ResolvedModel:
    local_path = model_store.ensure_model(spec=spec, offline=offline)
    manifest = model_store._load_manifest()
    repo_id = manifest.get("models", {}).get(spec.key, {}).get("repo_id", spec.repo_candidates[0])
    return ResolvedModel(model_key=spec.key, repo_id=repo_id, local_path=local_path)
