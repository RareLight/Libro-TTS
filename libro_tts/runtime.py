from __future__ import annotations

from dataclasses import dataclass, field, replace
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import json
import logging
import multiprocessing as mp
from numbers import Real
from pathlib import Path
import numpy as np
import re
import sys
from typing import Any

from .assets import AssetSpec, WHISPER, ensure_asset, local_asset_loading, prepare_model_assets
from huggingface_hub import constants

from .catalog import (
    DEFAULT_AUDIO_FORMAT,
    CATALOG,
    DEFAULT_MAX_TOKENS,
    ModelSpec,
    accepted_model_types_for_key,
    get_model_spec,
    is_mlx_repo_id,
)
from .env import validate_model_load_preflight, validate_local_encoder
from .progress import ChunkProgressBar
from .diagnostics import TailCapture
from .audio_output import AtomicAudioFile, StreamingWavOutput, AudioOutputError
from .paths import project_root
from .store import ModelStore, resolve_model
from .text import sanitize_text_for_tts
from .validation import (
    MAX_CHUNK_DURATION_SECONDS, BASE_CHARS_PER_SECOND,
    canonical_path_key,
    ENCODED_AUDIO_FORMATS, validate_audio_format, validate_speed,
    validate_input_path, validate_output_destination,
    resolve_output_path as _resolve_output_path,
)

SPARK_ALLOWED_SPEEDS = (0.0, 0.5, 1.0, 1.5, 2.0)
HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
VOICE_PROMPT_MODELS = frozenset(key for key, spec in CATALOG.items() if spec.voice_mode == "prompt")
REF_TEXT_PROMPT_MODELS = frozenset(key for key, spec in CATALOG.items() if spec.requires_ref_text)
REFERENCE_VOICE_DIR = "reference_voices"
DEFAULT_STT_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"
SPARK_MAX_REF_TEXT_CHARS = 320
KNOWN_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".opus")
_LOADED_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_LOADED_STT_MODEL: Any | None = None
_LOADED_STT_REFERENCE: str | None = None
_PARALLEL_WORKER_MODEL: Any | None = None
_PARALLEL_WORKER_MODEL_REFERENCE: tuple[str, str] | None = None
TARGET_CHUNK_DURATION_SECONDS = 12.5
MIN_CHUNK_DURATION_SECONDS = 10.0
MIN_CHARS_PER_CHUNK = 80
DEFAULT_PARALLEL_WORKERS = 2
DEFAULT_PARALLEL_MAX_CHUNKS = 12
MIN_PARALLEL_CHUNK_COUNT = 3
PARALLEL_DISABLED_MODEL_KEYS = frozenset(key for key, spec in CATALOG.items() if not spec.parallel_safe)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class RuntimeOptions:
    model_key: str
    model_path_override: str | None
    voice: str | None
    speed: float | None
    lang_code: str | None
    max_tokens: int | None
    verbose: bool
    offline: bool
    audio_format: str = DEFAULT_AUDIO_FORMAT
    input_encoding: str = "utf-8"
    input_encoding_fallbacks: tuple[str, ...] = ("utf-8-sig", "latin-1")
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    parallel_max_chunks: int = DEFAULT_PARALLEL_MAX_CHUNKS
    stream_wav: bool = True


@dataclass
class RuntimeResult:
    files_processed: int


@dataclass(frozen=True)
class BatchFailure:
    input_file: str
    error: str


@dataclass
class BatchRuntimeResult:
    total_files: int
    files_processed: int
    files_failed: int
    failures: list[BatchFailure] = field(default_factory=list)


def _get_load_model_fn():
    from mlx_audio.tts.utils import load_model

    return load_model


def _get_audio_write_fn():
    from mlx_audio.audio_io import write as audio_write

    return audio_write


def _get_stt_load_fn():
    from mlx_audio.stt import load

    return load


@local_asset_loading()
def _load_runtime_model(model_reference: str, logger: logging.Logger):
    cache_key = (model_reference, str(constants.HF_HUB_CACHE))
    cached = _LOADED_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    logger.info("Loading TTS model runtime from '%s'", model_reference)
    load_model = _get_load_model_fn()
    model = load_model(model_reference)
    _LOADED_MODEL_CACHE[cache_key] = model
    return model


def _print_user_message(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    print(message, file=stream, flush=True)


def _display_model_name(model_reference: str) -> str:
    if HF_REPO_ID_PATTERN.match(model_reference):
        return model_reference.split("/", 1)[-1]

    name = Path(model_reference).name
    if "__" in name:
        return name.split("__", 1)[-1]
    return name


def _normalize_speed(model_key: str | None, speed: float | None) -> float | None:
    validate_speed(model_key, speed)
    if speed is None:
        return None

    if model_key != "spark":
        return speed

    nearest = min(SPARK_ALLOWED_SPEEDS, key=lambda value: abs(value - speed))
    return float(nearest)


def _estimate_chunk_length_bounds(speed: float | None) -> tuple[int, int, int]:
    speed_factor = speed if speed and speed > 0 else 1.0
    chars_per_second = BASE_CHARS_PER_SECOND * speed_factor

    min_chars = max(int(chars_per_second * MIN_CHUNK_DURATION_SECONDS), MIN_CHARS_PER_CHUNK)
    target_chars = max(int(chars_per_second * TARGET_CHUNK_DURATION_SECONDS), min_chars)
    max_chars = max(int(chars_per_second * MAX_CHUNK_DURATION_SECONDS), target_chars)
    return min_chars, target_chars, max_chars


def _split_oversized_unit(unit: str, max_chars: int) -> list[str]:
    words = unit.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if word_len > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for index in range(0, word_len, max_chars):
                chunks.append(word[index : index + max_chars])
            continue

        candidate_len = word_len if current_len == 0 else current_len + 1 + word_len
        if candidate_len > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len = candidate_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _chunk_text_for_generation(text: str, speed: float | None) -> list[str]:
    if not text.strip():
        return []

    min_chars, target_chars, max_chars = _estimate_chunk_length_bounds(speed)
    base_units = [unit.strip() for unit in SENTENCE_SPLIT_PATTERN.split(text) if unit.strip()]
    if not base_units:
        return [text.strip()]

    units: list[str] = []
    for unit in base_units:
        if len(unit) <= max_chars:
            units.append(unit)
        else:
            units.extend(_split_oversized_unit(unit, max_chars))

    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = unit if not current else f"{current} {unit}"
        if len(candidate) <= target_chars:
            current = candidate
            continue

        if len(candidate) <= max_chars and len(current) < min_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
        current = unit

    if current:
        if chunks and len(current) < min_chars:
            merged = f"{chunks[-1]} {current}".strip()
            if len(merged) <= max_chars * 2:
                chunks[-1] = merged
            else:
                chunks.append(current.strip())
        else:
            chunks.append(current.strip())

    return chunks if chunks else [text.strip()]


def _resolve_effective_defaults(
    spec: ModelSpec | None,
    voice: str | None,
    speed: float | None,
    lang_code: str | None,
) -> tuple[str | None, float | None, str | None]:
    default_voice = spec.default_voice if spec else None
    default_speed = spec.default_speed if spec else None
    default_lang = spec.default_lang_code if spec else None

    effective_voice = voice if voice is not None else default_voice
    effective_speed = speed if speed is not None else default_speed
    effective_lang = lang_code if lang_code is not None else default_lang

    return effective_voice, effective_speed, effective_lang


def _resolve_voice_prompt_path(voice: str) -> str:
    candidate = Path(voice).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    prompt_name = voice if voice.lower().endswith(".wav") else f"{voice}.wav"
    prompt_path = project_root() / REFERENCE_VOICE_DIR / prompt_name
    if prompt_path.is_file():
        return str(prompt_path.resolve())

    raise FileNotFoundError(
        f"Voice prompt '{voice}' was not found. Provide a valid .wav file path "
        f"or a prompt name present in ./{REFERENCE_VOICE_DIR}."
    )


def _resolve_reference_text(ref_audio_path: str) -> str | None:
    ref_text_path = Path(ref_audio_path).with_suffix(".txt")
    if not ref_text_path.is_file():
        return None

    decode_errors: list[str] = []
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            content = _normalize_reference_text(ref_text_path.read_text(encoding=encoding))
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: byte {exc.start} ({exc.reason})")
            continue
        return content or None
    raise RuntimeError(
        f"Reference transcript exists but could not be decoded: {ref_text_path}. "
        f"Attempted encodings: utf-8-sig, latin-1. Errors: {' | '.join(decode_errors)}"
    )


def _normalize_reference_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lstrip().removeprefix("\ufeff")).strip()


def _truncate_reference_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    clipped = value[:max_chars].rstrip()
    punctuation_breaks = [clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"), clipped.rfind(";")]
    sentence_break = max(punctuation_breaks)
    if sentence_break >= max_chars // 2:
        return clipped[: sentence_break + 1].rstrip()
    return clipped


def _resolve_effective_voice(model_key: str | None, voice: str | None) -> str | None:
    if voice is None:
        return None

    if model_key in VOICE_PROMPT_MODELS:
        return _resolve_voice_prompt_path(voice)

    return voice


def _resolve_local_kokoro_voice(model_reference: str, voice: str, offline: bool,
                                model_store: ModelStore | None = None) -> str:
    """Use included voice tensors instead of requiring a second HF snapshot."""
    voices_dir = Path(model_reference) / "voices"
    resolved: list[str] = []
    for name in voice.split(","):
        candidate = Path(name).expanduser() if name.endswith(".safetensors") else voices_dir / f"{name}.safetensors"
        if not candidate.is_file() or candidate.stat().st_size == 0:
            if name.endswith(".safetensors") or model_store is None:
                raise RuntimeError(
                    f"Kokoro voice '{name}' is missing locally: {candidate}. "
                    "Supply a nonempty voice tensor or acquire the selected voice online."
                )
            if not re.fullmatch(r"[a-z]{2}_[A-Za-z0-9_]+", name):
                raise RuntimeError(f"Invalid Kokoro voice name: {name}")
            asset = f"voices/{name}.safetensors"
            snapshot = ensure_asset(model_store, AssetSpec(f"kokoro_voice_{name}", "prince-canuma/Kokoro-82M",
                                                           (asset,), (asset,)), offline)
            candidate = snapshot / asset
        resolved.append(str(candidate.resolve()))
    return ",".join(resolved)


def _validate_voxtral_voice(model_reference: str, voice: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", voice):
        raise RuntimeError(f"Invalid Voxtral voice name: {voice}")
    root = Path(model_reference).resolve()
    path = root / "voice_embedding" / f"{voice}.safetensors"
    if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Voxtral voice '{voice}' is missing a nonempty local embedding: {path}. "
                           "Repair the model snapshot or choose an included voice.")


def build_generate_kwargs(
    *,
    text: str,
    spec: ModelSpec | None,
    model_key: str | None = None,
    voice: str | None,
    speed: float | None,
    lang_code: str | None,
    max_tokens: int | None,
    verbose: bool,
) -> dict[str, Any]:
    resolved_model_key = spec.key if spec is not None else model_key
    effective_voice, effective_speed, effective_lang = _resolve_effective_defaults(
        spec=spec,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
    )
    effective_voice = _resolve_effective_voice(resolved_model_key, effective_voice)
    effective_speed = _normalize_speed(resolved_model_key, effective_speed)

    cleaned_text = sanitize_text_for_tts(text)

    kwargs: dict[str, Any] = {
        "text": cleaned_text,
        "verbose": verbose,
        "max_tokens": max_tokens if max_tokens is not None else (spec.max_tokens if spec else DEFAULT_MAX_TOKENS),
    }

    if effective_voice is not None and resolved_model_key in VOICE_PROMPT_MODELS:
        kwargs["ref_audio"] = effective_voice

    elif effective_voice is not None:
        kwargs["voice"] = effective_voice

    if effective_speed is not None:
        kwargs["speed"] = effective_speed
    if effective_lang is not None:
        kwargs["lang_code"] = effective_lang

    if spec:
        kwargs.update({k: v for k, v in spec.extra_kwargs.items() if v is not None})

    return kwargs


def _auto_transcribe_reference_text(
    *,
    ref_audio_path: str,
    logger: logging.Logger,
    verbose: bool,
    model_store: ModelStore | None = None,
    offline: bool = False,
) -> str:
    global _LOADED_STT_MODEL, _LOADED_STT_REFERENCE

    store = model_store if model_store is not None else ModelStore()
    local_path = str(ensure_asset(store, WHISPER, offline))
    with local_asset_loading(store.root_dir / ".hf"):
        if _LOADED_STT_MODEL is None or _LOADED_STT_REFERENCE != local_path:
            if verbose:
                logger.info("Loading local STT model '%s' for ref_text auto-transcription", local_path)
            _LOADED_STT_MODEL = _get_stt_load_fn()(local_path)
            _LOADED_STT_REFERENCE = local_path
        result = _LOADED_STT_MODEL.generate(ref_audio_path)
    transcript = result if isinstance(result, str) else getattr(result, "text", None)
    transcript = _normalize_reference_text(transcript) if isinstance(transcript, str) else ""
    if not transcript:
        raise RuntimeError(f"Auto-transcription returned empty or invalid text for '{ref_audio_path}'.")
    return transcript


@dataclass(frozen=True)
class ResolvedReferenceTranscript:
    text: str
    source: str


@dataclass
class ReferenceContext:
    """Successful transcript reuse scoped to one batch, never across runs."""
    transcripts: dict[tuple[Any, ...], ResolvedReferenceTranscript] = field(default_factory=dict)


def _reference_file_identity(path: Path) -> tuple[Any, ...]:
    try:
        info = path.stat()
        return (str(path.resolve()), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    except FileNotFoundError:
        return (str(path.resolve()), None)


def _populate_missing_ref_text(
    *,
    model_key: str,
    kwargs: dict[str, Any],
    logger: logging.Logger,
    verbose: bool,
    model_store: ModelStore | None = None,
    offline: bool = False,
    reference_context: ReferenceContext | None = None,
) -> ResolvedReferenceTranscript | None:
    if model_key not in REF_TEXT_PROMPT_MODELS:
        return
    ref_audio = kwargs.get("ref_audio")
    if ref_audio is None:
        return
    cache_key = None
    if reference_context is not None:
        explicit = kwargs.get("ref_text")
        cache_key = (model_key, str(model_store.root_dir.resolve()) if model_store else None, offline,
                     _reference_file_identity(Path(ref_audio)),
                     _reference_file_identity(Path(ref_audio).with_suffix(".txt")),
                     _normalize_reference_text(explicit) if isinstance(explicit, str) else None)
        cached = reference_context.transcripts.get(cache_key)
        if cached is not None:
            kwargs["ref_text"] = cached.text
            return cached
    ref_text_source: str
    ref_text: str | None = None

    existing_ref_text = kwargs.get("ref_text")
    if isinstance(existing_ref_text, str) and _normalize_reference_text(existing_ref_text):
        ref_text = existing_ref_text
        ref_text_source = "explicit"
    else:
        ref_text_from_file = _resolve_reference_text(str(ref_audio))
        if ref_text_from_file:
            ref_text = ref_text_from_file
            ref_text_source = "file"

    if ref_text is None:
        ref_text_path = Path(ref_audio).with_suffix(".txt")
        logger.warning(
            "Missing reference transcript for '%s'. Expected sibling text file '%s'; attempting auto-transcription.",
            ref_audio,
            ref_text_path,
        )
        ref_text = _auto_transcribe_reference_text(
            ref_audio_path=str(ref_audio),
            logger=logger,
            verbose=verbose,
            model_store=model_store,
            offline=offline,
        )
        ref_text_source = "stt"

    ref_text = _normalize_reference_text(ref_text)
    if not ref_text:
        raise RuntimeError(f"Resolved reference transcript is empty for '{ref_audio}'.")

    if model_key == "spark" and ref_text_source == "file" and len(ref_text) > SPARK_MAX_REF_TEXT_CHARS:
        logger.warning(
            "Spark reference transcript '%s' is long (%s chars). Auto-transcribing from '%s' for better prompt alignment.",
            Path(str(ref_audio)).with_suffix(".txt"),
            len(ref_text),
            ref_audio,
        )
        try:
            transcribed = _normalize_reference_text(
                _auto_transcribe_reference_text(
                    ref_audio_path=str(ref_audio),
                    logger=logger,
                    verbose=verbose,
                    model_store=model_store,
                    offline=offline,
                )
            )
            if not transcribed:
                raise RuntimeError("Auto-transcription returned empty text.")
            ref_text = transcribed
            ref_text_source = "stt"
        except Exception as exc:
            logger.warning(
                "Spark auto-transcription failed (%s). Falling back to trimmed transcript from file.",
                exc,
            )

    if model_key == "spark" and len(ref_text) > SPARK_MAX_REF_TEXT_CHARS:
        original_len = len(ref_text)
        ref_text = _truncate_reference_text(ref_text, SPARK_MAX_REF_TEXT_CHARS)
        logger.warning(
            "Trimmed Spark reference transcript from %s to %s chars to avoid oversized prompt context.",
            original_len,
            len(ref_text),
        )

    kwargs["ref_text"] = ref_text
    resolved = ResolvedReferenceTranscript(text=ref_text, source=ref_text_source)
    if reference_context is not None:
        reference_context.transcripts[cache_key] = resolved
    return resolved


def _validate_model_generate_kwargs(*, model_key: str, kwargs: dict[str, Any]) -> None:
    if model_key not in VOICE_PROMPT_MODELS:
        return

    has_ref_audio = kwargs.get("ref_audio") is not None
    has_voice = kwargs.get("voice") is not None

    if has_voice:
        raise RuntimeError(
            f"Model '{model_key}' cloning expects ref_audio (and sometimes ref_text); "
            "internal kwargs unexpectedly contained 'voice'."
        )

    if model_key in REF_TEXT_PROMPT_MODELS and has_ref_audio:
        ref_text = kwargs.get("ref_text")
        if not isinstance(ref_text, str) or not ref_text.strip():
            raise RuntimeError(
                f"Model '{model_key}' cloning requires reference text paired with ref audio. "
                "Provide <voice>.txt next to the .wav file in ./reference_voices, or allow auto-transcription."
            )


def _to_numpy_audio(audio: Any) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        array = audio
    elif hasattr(audio, "__array__"):
        array = np.asarray(audio)
    elif hasattr(audio, "tolist"):
        array = np.array(audio.tolist())
    else:
        array = np.array(audio)

    if array.ndim == 0:
        raise RuntimeError("Generated audio chunk is scalar; expected 1D waveform data.")

    if array.ndim == 2 and 1 in array.shape:
        # Preserve legacy row/column mono handling, including one sample.
        array = array.reshape(-1)
    elif array.ndim > 2:
        raise RuntimeError(
            f"Generated audio chunk has unsupported shape {array.shape}; expected mono or channel-last audio."
        )

    if array.dtype.kind not in "iuf":
        raise RuntimeError(f"Generated audio must contain real numeric samples; received {array.dtype}.")
    if not np.isfinite(array).all():
        raise RuntimeError("Generated audio contains non-finite samples (NaN or infinity).")
    # The pinned encoder handles float32/float64 as normalized PCM; float16
    # would otherwise be cast directly to int16, losing fractional samples.
    if array.dtype == np.float16:
        array = array.astype(np.float32)
    return array


def _validate_sample_rate(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise RuntimeError(f"Generated audio sample rate must be a positive integer; received {value!r}.")
    try:
        rate = int(value)
    except (ValueError, OverflowError) as exc:
        raise RuntimeError(f"Generated audio sample rate must be a positive integer; received {value!r}.") from exc
    if rate <= 0 or rate != value:
        raise RuntimeError(f"Generated audio sample rate must be a positive integer; received {value!r}.")
    return rate


def _validate_channel_layout(chunks: list[np.ndarray], chunk: np.ndarray) -> None:
    if chunks and chunk.shape[1:] != chunks[0].shape[1:]:
        raise RuntimeError(
            "Inconsistent channel layouts across generated chunks: "
            f"{chunks[0].shape} vs {chunk.shape}; expected consistent mono or channel-last audio."
        )


def _validate_audio_consistency(
    chunks: list[np.ndarray], chunk: np.ndarray, sample_rate: int, expected_sample_rate: int | None,
) -> int:
    rate = _validate_sample_rate(sample_rate)
    if expected_sample_rate is not None and rate != expected_sample_rate:
        raise RuntimeError(f"Inconsistent sample rates across generated chunks: {expected_sample_rate} vs {rate}")
    _validate_channel_layout(chunks, chunk)
    return rate


def _collect_generation_audio(
    results,
    fallback_sample_rate: int | None,
    progress: ChunkProgressBar | None = None,
) -> tuple[list[np.ndarray], int]:
    sample_rate = _validate_sample_rate(fallback_sample_rate) if fallback_sample_rate is not None else None
    chunks: list[np.ndarray] = []

    for result in results:
        if result is None:
            continue
        audio = getattr(result, "audio", None)
        if audio is None:
            continue

        chunk = _to_numpy_audio(audio)
        if chunk.size == 0:
            continue

        result_sample_rate = getattr(result, "sample_rate", None)
        if result_sample_rate is None:
            result_sample_rate = sample_rate
        if result_sample_rate is not None:
            sample_rate = _validate_audio_consistency(chunks, chunk, result_sample_rate, sample_rate)
        else:
            _validate_channel_layout(chunks, chunk)
        chunks.append(chunk)
        if progress is not None:
            progress.update(1)

    if not chunks:
        raise RuntimeError("Model generation completed without any audio chunks.")
    if sample_rate is None:
        raise RuntimeError("Model generation did not expose a sample rate.")

    return chunks, sample_rate


def _model_supports_parallel_chunk_generation(model_key: str | None) -> bool:
    if model_key is None:
        return True
    return model_key not in PARALLEL_DISABLED_MODEL_KEYS


def _should_parallelize_chunk_generation(
    options: RuntimeOptions,
    chunk_count: int,
    model_key: str | None = None,
) -> bool:
    if not _model_supports_parallel_chunk_generation(model_key):
        return False
    workers = max(int(options.parallel_workers), 1)
    if workers < 2:
        return False
    if chunk_count < MIN_PARALLEL_CHUNK_COUNT:
        return False
    return True


def _parallel_chunk_batch_size(options: RuntimeOptions, chunk_count: int) -> int:
    configured = int(options.parallel_max_chunks)
    if configured <= 0:
        return max(chunk_count, 1)
    return max(min(configured, chunk_count), 1)


@local_asset_loading()
def _generate_chunk_audio(
    *,
    model: Any,
    kwargs_template: dict[str, Any],
    text_chunk: str,
    suppress_model_output: bool = False,
) -> tuple[np.ndarray, int]:
    chunk_kwargs = dict(kwargs_template)
    chunk_kwargs["text"] = text_chunk
    if suppress_model_output:
        with TailCapture() as sink, redirect_stdout(sink), redirect_stderr(sink):
            try:
                chunk_results = model.generate(**chunk_kwargs)
                chunk_segments, chunk_sample_rate = _collect_generation_audio(
                    results=chunk_results,
                    fallback_sample_rate=getattr(model, "sample_rate", None),
                )
            except Exception as exc:
                tail = sink.getvalue().strip()
                if tail:
                    raise RuntimeError(f"{exc}\nRecent model output (tail):\n{tail}") from exc
                raise
    else:
        chunk_results = model.generate(**chunk_kwargs)
        chunk_segments, chunk_sample_rate = _collect_generation_audio(
            results=chunk_results,
            fallback_sample_rate=getattr(model, "sample_rate", None),
        )
    chunk_audio = chunk_segments[0] if len(chunk_segments) == 1 else np.concatenate(chunk_segments, axis=0)
    return chunk_audio, chunk_sample_rate


def _parallel_worker_generate(
    task: tuple[int, str, str, dict[str, Any], bool, str]
) -> tuple[int, np.ndarray, int]:
    global _PARALLEL_WORKER_MODEL, _PARALLEL_WORKER_MODEL_REFERENCE
    index, text_chunk, model_reference, kwargs_template, suppress_model_output, cache_root = task

    with local_asset_loading(Path(cache_root)):
        cache_key = (model_reference, str(constants.HF_HUB_CACHE))
        if _PARALLEL_WORKER_MODEL is None or _PARALLEL_WORKER_MODEL_REFERENCE != cache_key:
            _PARALLEL_WORKER_MODEL = _load_runtime_model(
                model_reference,
                logging.getLogger("libro_tts.parallel"),
            )
            _PARALLEL_WORKER_MODEL_REFERENCE = cache_key

        chunk_audio, chunk_sample_rate = _generate_chunk_audio(
            model=_PARALLEL_WORKER_MODEL,
            kwargs_template=kwargs_template,
            text_chunk=text_chunk,
            suppress_model_output=suppress_model_output,
        )
        return index, chunk_audio, chunk_sample_rate


def _generate_audio_serial(
    *,
    model: Any,
    kwargs_template: dict[str, Any],
    text_chunks: list[str],
    progress: ChunkProgressBar | None = None,
    suppress_model_output: bool = False,
    audio_sink: StreamingWavOutput | None = None,
) -> tuple[list[np.ndarray], int]:
    combined_audio: list[np.ndarray] = []
    combined_sample_rate: int | None = None
    accepted_chunks = 0

    for text_chunk in text_chunks:
        chunk_audio, chunk_sample_rate = _generate_chunk_audio(
            model=model,
            kwargs_template=kwargs_template,
            text_chunk=text_chunk,
            suppress_model_output=suppress_model_output,
        )
        combined_sample_rate = _validate_audio_consistency(
            combined_audio, chunk_audio, chunk_sample_rate, combined_sample_rate,
        )
        if audio_sink is not None:
            audio_sink.append(chunk_audio, chunk_sample_rate)
        else:
            combined_audio.append(chunk_audio)
        accepted_chunks += 1
        if progress is not None:
            progress.update(1)

    if combined_sample_rate is None or not accepted_chunks:
        raise RuntimeError("No audio returned from synthesis.")

    return combined_audio, combined_sample_rate


def _generate_audio_parallel(
    *,
    model_reference: str,
    kwargs_template: dict[str, Any],
    text_chunks: list[str],
    workers: int,
    batch_size: int,
    progress: ChunkProgressBar | None = None,
    suppress_model_output: bool = False,
    audio_sink: StreamingWavOutput | None = None,
) -> tuple[list[np.ndarray], int]:
    context = mp.get_context("spawn")
    combined_audio: list[np.ndarray] = []
    combined_sample_rate: int | None = None
    accepted_chunks = 0
    with context.Pool(processes=workers) as pool:
        for start in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[start : start + batch_size]
            indexed_tasks = [
                (start + offset, chunk, model_reference, kwargs_template, suppress_model_output, str(constants.HF_HOME))
                for offset, chunk in enumerate(batch_chunks)
            ]
            batch_results = pool.map(_parallel_worker_generate, indexed_tasks)
            batch_results.sort(key=lambda item: item[0])

            for _, chunk_audio, chunk_sample_rate in batch_results:
                combined_sample_rate = _validate_audio_consistency(
                    combined_audio, chunk_audio, chunk_sample_rate, combined_sample_rate,
                )
                if audio_sink is not None:
                    audio_sink.append(chunk_audio, chunk_sample_rate)
                else:
                    combined_audio.append(chunk_audio)
                accepted_chunks += 1
                if progress is not None:
                    progress.update(1)
            del batch_results

    if combined_sample_rate is None or not accepted_chunks:
        raise RuntimeError("No audio returned from synthesis.")

    return combined_audio, combined_sample_rate


def _normalized_output_stem(input_file: Path) -> str:
    stem = input_file.stem
    stem_lower = stem.lower()
    for suffix in KNOWN_AUDIO_SUFFIXES:
        if stem_lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    stem = stem.lstrip(".")
    if not stem:
        return "untitled"
    return stem


def default_output_target_for_input(
    *,
    input_file: Path,
    audio_format: str,
    output_dir: Path | None = None,
) -> str:
    stem = _normalized_output_stem(input_file)
    filename = f"{stem}.{audio_format.lstrip('.')}"
    if output_dir is None:
        return str(input_file.with_name(filename))
    return str(output_dir / filename)


def _plan_batch_outputs(
    files: list[Path], output_dir: Path, audio_format: str,
) -> list[tuple[Path, Path]]:
    plan: list[tuple[Path, Path]] = []
    destinations: dict[str, list[tuple[Path, Path]]] = {}
    for input_file in files:
        target = default_output_target_for_input(
            input_file=input_file, audio_format=audio_format, output_dir=output_dir,
        )
        output_path = _resolve_output_path(target, audio_format)
        # Treat case and canonical Unicode variants as ambiguous even on a
        # case-sensitive volume. Resolve existing output symlinks as writes do.
        key = canonical_path_key(output_path)
        entry = (input_file, output_path)
        destinations.setdefault(key, []).append(entry)
        plan.append(entry)

    collisions = [entries for entries in destinations.values() if len(entries) > 1]
    if collisions:
        details = "\n".join(
            "  " + ", ".join(repr(source.name) for source, _ in entries)
            + f" -> {entries[0][1]}"
            for entries in collisions
        )
        raise ValueError(
            "Batch output collisions detected; no files were processed. "
            "Rename the conflicting inputs or place them in separate input directories:\n" + details
        )
    return plan


def _build_encoding_order(options: RuntimeOptions) -> list[str]:
    ordered: list[str] = []
    for encoding in (options.input_encoding, *options.input_encoding_fallbacks):
        normalized = encoding.strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)

    if not ordered:
        ordered.append("utf-8")
    return ordered


def _read_input_text(
    *,
    input_file: Path,
    options: RuntimeOptions,
    logger: logging.Logger,
) -> str:
    decode_errors: list[str] = []
    encodings = _build_encoding_order(options)

    for index, encoding in enumerate(encodings):
        try:
            text = input_file.read_text(encoding=encoding)
        except LookupError as exc:
            raise ValueError(f"Unknown text encoding '{encoding}'.") from exc
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: byte {exc.start} ({exc.reason})")
            continue

        if index > 0:
            warning = f"Decoded '{input_file.name}' using fallback encoding '{encoding}'."
            if options.verbose:
                logger.warning(warning)
            else:
                _print_user_message(warning, error=True)
        return text.removeprefix("\ufeff")

    raise RuntimeError(
        f"Failed to decode input file '{input_file}'. Attempted encodings: {', '.join(encodings)}. "
        f"Errors: {' | '.join(decode_errors)}"
    )


def _write_output_audio(
    *,
    output_target: str,
    audio_format: str,
    audio: np.ndarray,
    sample_rate: int,
) -> Path:
    output_path = _resolve_output_path(output_target, audio_format)
    try:
        with AtomicAudioFile(output_path) as transaction:
            _get_audio_write_fn()(str(transaction.temporary), audio, sample_rate, format=audio_format)
            return transaction.publish()
    except OSError as exc:
        raise RuntimeError(f"Failed to write audio output '{output_path}': {exc}") from exc


def _validate_local_model_path(path: Path, model_key: str, logger: logging.Logger) -> None:
    if not path.is_dir():
        raise RuntimeError(
            f"--model-path must point to a model directory. Received non-directory path: {path}"
        )

    config_path = path / "config.json"
    if not config_path.exists():
        raise RuntimeError(
            f"--model-path is missing required config.json: {config_path}"
        )

    if not any(path.glob("**/*.safetensors")):
        raise RuntimeError(
            f"--model-path does not contain any .safetensors weights: {path}"
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(config, dict):
        return

    model_type = config.get("model_type")
    model_type_normalized = (
        str(model_type).strip().lower() if isinstance(model_type, str) else None
    )
    accepted_model_types = accepted_model_types_for_key(model_key)
    if model_type_normalized and model_type_normalized not in accepted_model_types:
        logger.warning(
            "Local model_type '%s' does not match selected --model '%s'. "
            "Proceeding with explicit user-provided --model-path.",
            model_type,
            model_key,
        )


def _resolve_model_reference(
    model_store: ModelStore,
    options: RuntimeOptions,
    logger: logging.Logger,
) -> tuple[str, ModelSpec | None]:
    if options.model_path_override:
        override = options.model_path_override
        path = Path(override)
        if path.exists():
            _validate_local_model_path(path, options.model_key, logger)
            return str(path.resolve()), None

        if HF_REPO_ID_PATTERN.match(override):
            if not is_mlx_repo_id(override):
                raise RuntimeError(
                    "--model-path must reference an MLX repo id (mlx-community/*) or an existing local path."
                )
            override_spec = replace(get_model_spec(options.model_key), repo_candidates=(override,))
            local = model_store.ensure_model(spec=override_spec, offline=options.offline)
            return str(local), None

        raise FileNotFoundError(
            "--model-path did not resolve to an existing local path. "
            "Provide a valid local path or an MLX repo id in the form mlx-community/<repo>."
        )

    spec = get_model_spec(options.model_key)
    resolved = resolve_model(model_store=model_store, spec=spec, offline=options.offline)
    logger.info(
        "Using model '%s' resolved to '%s' at '%s'",
        spec.key,
        resolved.repo_id,
        resolved.local_path,
    )
    return str(resolved.local_path), spec


def generate_tts(
    *,
    text: str,
    output_prefix: str,
    model_store: ModelStore,
    options: RuntimeOptions,
    logger: logging.Logger,
    reference_context: ReferenceContext | None = None,
) -> None:
    validate_audio_format(options.audio_format)
    validate_speed(options.model_key, options.speed)
    cleaned_text = sanitize_text_for_tts(text.removeprefix("\ufeff"))
    if not cleaned_text.strip():
        raise ValueError("Input contains no text for synthesis after preprocessing.")
    validate_output_destination(_resolve_output_path(output_prefix, options.audio_format))
    if options.audio_format.lower() in ENCODED_AUDIO_FORMATS:
        validate_local_encoder()
    # Resolve cheap voice/prompt/default checks before downloading model assets.
    requested_spec = None if options.model_path_override else get_model_spec(options.model_key)
    kwargs = build_generate_kwargs(
        text=cleaned_text, spec=requested_spec, model_key=options.model_key,
        voice=options.voice, speed=options.speed, lang_code=options.lang_code,
        max_tokens=options.max_tokens, verbose=options.verbose,
    )
    model_reference, spec = _resolve_model_reference(
        model_store=model_store,
        options=options,
        logger=logger,
    )
    resolved_model_key = spec.key if spec is not None else options.model_key
    validate_model_load_preflight(resolved_model_key, model_reference)
    prepare_model_assets(model_store, resolved_model_key, model_reference, options.offline)

    voice_label = kwargs.get("voice")
    if resolved_model_key == "kokoro" and isinstance(voice_label, str):
        kwargs["voice"] = _resolve_local_kokoro_voice(model_reference, voice_label, options.offline, model_store)
    if resolved_model_key == "voxtral_tts":
        _validate_voxtral_voice(model_reference, str(voice_label or "casual_male"))
    try:
        reference = _populate_missing_ref_text(
            model_key=resolved_model_key,
            kwargs=kwargs,
            logger=logger,
            verbose=options.verbose,
            model_store=model_store,
            offline=options.offline,
            reference_context=reference_context,
        )
        if reference is not None:
            logger.info("Using %s reference transcript for '%s'", reference.source, kwargs.get("ref_audio"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to prepare reference transcript for model '{resolved_model_key}': {exc}"
        ) from exc
    _validate_model_generate_kwargs(model_key=resolved_model_key, kwargs=kwargs)

    with local_asset_loading(model_store.root_dir / ".hf"), ExitStack() as output_stack:
        audio_sink = output_stack.enter_context(StreamingWavOutput(_resolve_output_path(output_prefix, options.audio_format))) if options.stream_wav and options.audio_format.lower() == "wav" else None
        text_to_generate = kwargs.pop("text", "")
        text_chunks = _chunk_text_for_generation(text_to_generate, kwargs.get("speed"))
        if not text_chunks:
            raise RuntimeError("No text available for synthesis after preprocessing.")

        if not options.verbose:
            if voice_label is None and kwargs.get("ref_audio") is not None:
                voice_label = Path(str(kwargs["ref_audio"])).name
            _print_user_message(
                "Model: "
                f"{_display_model_name(model_reference)} | "
                f"voice={voice_label or 'default'} | "
                f"speed={kwargs.get('speed', 'default')} | "
                f"lang={kwargs.get('lang_code', 'default')}"
            )
        model_parallel_supported = _model_supports_parallel_chunk_generation(resolved_model_key)
        use_parallel = _should_parallelize_chunk_generation(
            options,
            len(text_chunks),
            model_key=resolved_model_key,
        )
        workers = max(int(options.parallel_workers), 1)
        batch_size = _parallel_chunk_batch_size(options, len(text_chunks))

        if not options.verbose and len(text_chunks) > 1:
            if use_parallel:
                _print_user_message(
                    f"Chunks: {len(text_chunks)} | mode=parallel | workers={workers} | batch={batch_size}"
                )
            elif not model_parallel_supported:
                _print_user_message(
                    f"Chunks: {len(text_chunks)} | mode=serial ({resolved_model_key} reliability guard)"
                )
            else:
                _print_user_message(f"Chunks: {len(text_chunks)} | mode=serial")

        progress = ChunkProgressBar(
            desc=f"Generating {_display_model_name(model_reference)}",
            enabled=not options.verbose,
            total=len(text_chunks),
        )

        try:
            if use_parallel:
                try:
                    combined_audio, combined_sample_rate = _generate_audio_parallel(
                        model_reference=model_reference,
                        kwargs_template=kwargs,
                        text_chunks=text_chunks,
                        workers=workers,
                        batch_size=batch_size,
                        progress=progress,
                        suppress_model_output=not options.verbose,
                        audio_sink=audio_sink,
                    )
                except AudioOutputError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Parallel chunk synthesis failed (%s). Falling back to serial mode.",
                        exc,
                    )
                    if not options.verbose:
                        _print_user_message(
                            "Parallel chunk synthesis failed; retrying in serial mode.",
                            error=True,
                        )
                    progress.reset()
                    if audio_sink is not None:
                        audio_sink.reset()
                    model = _load_runtime_model(model_reference, logger)
                    combined_audio, combined_sample_rate = _generate_audio_serial(
                        model=model,
                        kwargs_template=kwargs,
                        text_chunks=text_chunks,
                        progress=progress,
                        suppress_model_output=not options.verbose,
                        audio_sink=audio_sink,
                    )
            else:
                model = _load_runtime_model(model_reference, logger)
                combined_audio, combined_sample_rate = _generate_audio_serial(
                    model=model,
                    kwargs_template=kwargs,
                    text_chunks=text_chunks,
                    progress=progress,
                    suppress_model_output=not options.verbose,
                    audio_sink=audio_sink,
                )
        finally:
            progress.close()

        if audio_sink is not None:
            output_path = audio_sink.publish()
        else:
            audio = combined_audio[0] if len(combined_audio) == 1 else np.concatenate(combined_audio, axis=0)
            output_path = _write_output_audio(
                output_target=output_prefix, audio_format=options.audio_format,
                audio=audio, sample_rate=combined_sample_rate,
            )
        logger.info("Saved generated audio to '%s'", output_path)
        if not options.verbose:
            _print_user_message(f"Saved: {output_path}")


def process_single_file(
    *,
    input_file: Path,
    output_prefix: str,
    model_store: ModelStore,
    options: RuntimeOptions,
    logger: logging.Logger,
    prepared_text: str | None = None,
    reference_context: ReferenceContext | None = None,
) -> RuntimeResult:
    validate_audio_format(options.audio_format)
    validate_speed(options.model_key, options.speed)
    validate_input_path(input_file)
    validate_output_destination(_resolve_output_path(output_prefix, options.audio_format),
                                protected_inputs=frozenset({canonical_path_key(input_file)}))

    text = prepared_text if prepared_text is not None else _read_input_text(
        input_file=input_file,
        options=options,
        logger=logger,
    )
    generate_tts(
        text=text,
        output_prefix=output_prefix,
        model_store=model_store,
        options=options,
        logger=logger,
        reference_context=reference_context,
    )
    return RuntimeResult(files_processed=1)


def process_batch_dir(
    *,
    input_dir: Path,
    output_dir: Path,
    model_store: ModelStore,
    options: RuntimeOptions,
    logger: logging.Logger,
) -> BatchRuntimeResult:
    validate_input_path(input_dir, directory=True)
    validate_audio_format(options.audio_format)
    validate_speed(options.model_key, options.speed)

    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.name.lower().endswith(".txt")
    )
    plan = _plan_batch_outputs(files, output_dir, options.audio_format)
    validate_output_destination(output_dir, directory=True)
    protected_inputs = frozenset(canonical_path_key(path) for path in files)
    for _, output_path in plan:
        validate_output_destination(output_path, protected_inputs=protected_inputs)
    if options.audio_format.lower() in ENCODED_AUDIO_FORMATS:
        validate_local_encoder()
    output_dir.mkdir(parents=True, exist_ok=True)
    files_processed = 0
    failures: list[BatchFailure] = []
    reference_context = ReferenceContext()
    for index, (input_file, output_path) in enumerate(plan, start=1):
        if not options.verbose:
            _print_user_message(f"[{index}/{len(files)}] {input_file.name}")
        try:
            process_single_file(
                input_file=input_file,
                output_prefix=str(output_path),
                model_store=model_store,
                options=options,
                logger=logger,
                reference_context=reference_context,
            )
            files_processed += 1
            logger.info("Processed: %s", input_file.name)
        except Exception as exc:
            failures.append(BatchFailure(input_file=input_file.name, error=str(exc)))
            if options.verbose:
                logger.error("Error processing '%s': %s", input_file.name, exc)
            else:
                _print_user_message(f"Failed: {input_file.name} ({exc})", error=True)

    return BatchRuntimeResult(
        total_files=len(files),
        files_processed=files_processed,
        files_failed=len(failures),
        failures=failures,
    )
