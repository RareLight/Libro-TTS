from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL_KEY = "kokoro"
DEFAULT_VOICE = "af_aoede"
DEFAULT_PROMPT_VOICE = "Britney.wav"
DEFAULT_SPEED = 1.1
DEFAULT_LANG_CODE = "a"
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_MAX_TOKENS = 1400
KOKORO_ENGLISH_VOICES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
)
VOXTRAL_ENGLISH_VOICES = (
    "casual_male",
    "casual_female",
    "cheerful_female",
    "neutral_male",
    "neutral_female",
)

REQUIRED_MODEL_KEYS = (
    "kokoro",
    "qwen3_tts",
    "csm",
    "dia",
    "spark",
    "chatterbox",
    "soprano",
    "voxtral_tts",
)

MLX_REPO_PREFIX = "mlx-community/"
MODEL_TYPE_CANONICAL: dict[str, str] = {
    # Older mlx-audio releases resolve CSM through the sesame architecture.
    "csm": "sesame",
}
MODEL_TYPE_ACCEPTED: dict[str, tuple[str, ...]] = {
    "csm": ("sesame", "csm"),
}
# Known backbone labels in the catalog's pinned TTS snapshots, not arbitrary
# architectures to relabel. Unexpected metadata must fail without mutation.
MODEL_CONFIG_TYPE_ADAPTERS = {"spark": ("qwen2",), "voxtral_tts": ("llama",), "csm": ("csm",)}
MODEL_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "kokoro": ("kokoro",),
    "qwen3_tts": ("qwen3_tts", "qwen3-tts", "qwen3tts", "qwen3", "qwen"),
    "csm": ("csm", "sesame", "marvis"),
    "dia": ("dia",),
    "spark": ("spark", "spark_tts", "spark-tts"),
    "chatterbox": ("chatterbox", "chatterbox_turbo", "chatterbox-turbo"),
    "soprano": ("soprano",),
    "voxtral_tts": ("voxtral_tts", "voxtral-tts", "voxtral"),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    repo_candidates: tuple[str, ...]
    default_voice: str | None
    default_speed: float | None
    default_lang_code: str | None
    max_tokens: int = DEFAULT_MAX_TOKENS
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    # Primary-snapshot assets used by the pinned mlx-audio runtime. Separate
    # HF repositories (codecs/STT) are not covered by these checks.
    required_assets: tuple[str, ...] = ()
    voice_mode: str = "named"
    requires_ref_text: bool = False
    parallel_safe: bool = True
    speed_behavior: str = "chunking only"


CATALOG: dict[str, ModelSpec] = {
    "kokoro": ModelSpec(
        key="kokoro",
        display_name="Kokoro 82M",
        repo_candidates=("mlx-community/Kokoro-82M-bf16",),
        default_voice=DEFAULT_VOICE,
        default_speed=DEFAULT_SPEED,
        default_lang_code="a",
        required_assets=("voices/af_aoede.safetensors",),
        speed_behavior="speech rate and chunking",
    ),
    "qwen3_tts": ModelSpec(
        key="qwen3_tts",
        display_name="Qwen3-TTS 1.7B Base",
        repo_candidates=(
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        ),
        default_voice="Vivian",
        default_speed=1.0,
        default_lang_code="english",
        max_tokens=4096,
        required_assets=(
            "tokenizer_config.json", "vocab.json", "merges.txt",
            "speech_tokenizer/config.json", "speech_tokenizer/*.safetensors",
        ),
    ),
    "csm": ModelSpec(
        key="csm",
        display_name="CSM 1B",
        repo_candidates=("mlx-community/csm-1b",),
        default_voice=DEFAULT_PROMPT_VOICE,
        default_speed=1.0,
        default_lang_code="en",
        voice_mode="prompt", requires_ref_text=True, parallel_safe=False,
    ),
    "dia": ModelSpec(
        key="dia",
        display_name="Dia 1.6B",
        repo_candidates=("mlx-community/Dia-1.6B-fp16",),
        default_voice=DEFAULT_PROMPT_VOICE,
        default_speed=1.0,
        default_lang_code="en",
        max_tokens=2000,
        voice_mode="prompt", requires_ref_text=True, parallel_safe=False,
    ),
    "spark": ModelSpec(
        key="spark",
        display_name="Spark-TTS 0.5B",
        repo_candidates=("mlx-community/Spark-TTS-0.5B-bf16",),
        default_voice=DEFAULT_PROMPT_VOICE,
        default_speed=1.0,
        default_lang_code="en",
        voice_mode="prompt", requires_ref_text=True,
        speed_behavior="categorical for unprompted speech; chunking only when cloning",
        required_assets=(
            "tokenizer.json", "tokenizer_config.json", "audio_tokenizer_config.yaml",
            "BiCodec/config.yaml", "BiCodec/*.safetensors",
            "wav2vec2-large-xlsr-53/config.json",
            "wav2vec2-large-xlsr-53/preprocessor_config.json",
            "wav2vec2-large-xlsr-53/*.safetensors",
        ),
    ),
    "chatterbox": ModelSpec(
        key="chatterbox",
        display_name="Chatterbox",
        repo_candidates=("mlx-community/chatterbox-fp16",),
        default_voice=DEFAULT_PROMPT_VOICE,
        default_speed=1.0,
        default_lang_code="en",
        required_assets=("tokenizer.json",),
        voice_mode="prompt",
    ),
    "soprano": ModelSpec(
        key="soprano",
        display_name="Soprano 80M",
        repo_candidates=("mlx-community/Soprano-1.1-80M-bf16",),
        default_voice=None,
        default_speed=1.0,
        default_lang_code="en",
        required_assets=("tokenizer.json", "tokenizer_config.json"),
        voice_mode="ignored",
    ),
    "voxtral_tts": ModelSpec(
        key="voxtral_tts",
        display_name="Voxtral 4B TTS",
        repo_candidates=("mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",),
        default_voice="neutral_female",
        default_speed=None,
        default_lang_code=None,
        max_tokens=4096,
        required_assets=("tekken.json", "voice_embedding/neutral_female.safetensors"),
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return CATALOG[model_key]
    except KeyError as exc:
        supported = ", ".join(sorted(CATALOG))
        raise ValueError(f"Unsupported model key '{model_key}'. Supported: {supported}") from exc


def list_model_keys() -> list[str]:
    return sorted(CATALOG.keys())


def list_model_specs() -> list[ModelSpec]:
    return [CATALOG[key] for key in list_model_keys()]


def is_mlx_repo_id(repo_id: str) -> bool:
    return repo_id.startswith(MLX_REPO_PREFIX)


def normalize_model_alias(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_model_key(value: str) -> str:
    normalized = normalize_model_alias(value)
    for key in list_model_keys():
        aliases = MODEL_KEY_ALIASES.get(key, (key,))
        if normalized in {normalize_model_alias(alias) for alias in aliases}:
            return key
    supported = ", ".join(list_model_keys())
    raise ValueError(f"Unsupported model '{value}'. Supported model keys: {supported}")


def canonical_model_type_for_key(model_key: str) -> str:
    return MODEL_TYPE_CANONICAL.get(model_key, model_key)


def accepted_model_types_for_key(model_key: str) -> tuple[str, ...]:
    accepted = MODEL_TYPE_ACCEPTED.get(model_key)
    if accepted is not None:
        return accepted
    canonical = canonical_model_type_for_key(model_key)
    if canonical == model_key:
        return (canonical,)
    return (canonical, model_key)


def list_kokoro_english_voices() -> tuple[str, ...]:
    return KOKORO_ENGLISH_VOICES


def list_voxtral_english_voices() -> tuple[str, ...]:
    return VOXTRAL_ENGLISH_VOICES
