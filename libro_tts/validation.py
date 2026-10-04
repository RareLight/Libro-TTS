"""Cheap request checks shared by CLI and runtime, without backend imports."""
from __future__ import annotations

import math
import os
from pathlib import Path
import unicodedata

SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "flac", "ogg", "opus", "vorbis", "pcm", "raw")
ENCODED_AUDIO_FORMATS = frozenset({"mp3", "flac", "ogg", "opus", "vorbis"})
MAX_CHUNK_DURATION_SECONDS = 15.0
BASE_CHARS_PER_SECOND = 12.0


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError("Value must be a positive integer.")
    return result


def validate_audio_format(value: str) -> str:
    if value.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format '{value}'. Choose: {', '.join(SUPPORTED_AUDIO_FORMATS)}.")
    return value


def validate_speed(model_key: str | None, speed: float | None) -> None:
    if speed is None:
        return
    if not math.isfinite(speed) or speed < 0 or speed == 0 and model_key != "spark":
        requirement = "finite and nonnegative" if model_key == "spark" else "finite and positive"
        raise ValueError(f"Speed for '{model_key or 'this model'}' must be {requirement}.")
    # Spark rounds to its existing discrete categories. Other speeds also drive
    # chunk bounds; reject values whose duration calculation would overflow.
    if model_key != "spark" and not math.isfinite(speed * BASE_CHARS_PER_SECOND * MAX_CHUNK_DURATION_SECONDS):
        raise ValueError("Speed is too large for synthesis chunk sizing.")


def validate_input_path(path: Path, *, directory: bool = False) -> None:
    expected = "directory" if directory else "file"
    if not path.exists():
        raise FileNotFoundError(f"Input {expected} does not exist: {path}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Input must be a regular {expected}: {path}")


def canonical_path_key(path: Path) -> str:
    """Conservatively match case/Unicode aliases, including output symlinks."""
    return unicodedata.normalize("NFC", str(path.resolve()).casefold())


def validate_output_destination(path: Path, *, directory: bool = False,
                                protected_inputs: frozenset[str] = frozenset()) -> None:
    """Check without creating files; protected inputs are canonical path keys."""
    try:
        destination = path.resolve()
        if canonical_path_key(destination) in protected_inputs:
            raise ValueError(f"Output would overwrite an input file: {path}")
        if destination.exists() and not (destination.is_dir() if directory else destination.is_file()):
            expected = "directory" if directory else "regular file"
            raise ValueError(f"Output must be a {expected}: {path}")
        parent = destination if directory else destination.parent
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError(f"Output parent is not a directory: {parent}")
        if not os.access(parent, os.W_OK | os.X_OK):
            raise ValueError(f"Output directory is not writable: {parent}")
    except OSError as exc:
        raise ValueError(f"Cannot access output destination '{path}': {exc}") from exc


def resolve_output_path(output_target: str, audio_format: str) -> Path:
    path = Path(output_target)
    target_suffix = f".{audio_format.lstrip('.')}"
    if path.suffix.lower() == target_suffix.lower():
        return path
    return path.with_suffix(target_suffix)
