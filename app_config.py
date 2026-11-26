"""Shared version and supported CPU Whisper models."""
VERSION = "0.3.1"
# Repository IDs match the aliases in Faster-Whisper 1.2.1.
# Publisher means the Hugging Face repository owner, not the model developer.
MODEL_REPOSITORIES = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "medium.en": "Systran/faster-whisper-medium.en",
    "small.en": "Systran/faster-whisper-small.en",
    "base.en": "Systran/faster-whisper-base.en",
    "tiny.en": "Systran/faster-whisper-tiny.en",
}
MODELS = (
    ("large-v3", "large-v3 - Systran"),
    ("large-v3-turbo", "large-v3-turbo - Mobius Labs"),
    ("distil-large-v3.5", "distil-large-v3.5 - Distil-Whisper"),
    ("distil-large-v3", "distil-large-v3 - Systran"),
    ("medium.en", "medium.en - Systran"),
    ("small.en", "small.en - Systran"),
    ("base.en", "base.en - Systran"),
    ("tiny.en", "tiny.en - Systran"),
)
DEFAULT_MODEL = "large-v3"
MODEL_IDS = frozenset(model for model, _ in MODELS)

# These are separate MLX weights, not the CPU CTranslate2 models.
MLX_REPOSITORIES = {
    "mlx-large-v3": "mlx-community/whisper-large-v3-mlx",
    "mlx-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
MLX_MODELS = (
    ("mlx-large-v3", "large-v3 - MLX Community"),
    ("mlx-large-v3-turbo", "large-v3-turbo - MLX Community"),
)
ALL_MODEL_IDS = MODEL_IDS | frozenset(MLX_REPOSITORIES)

# Approximate quality-first ordering, not a measured ranking.
MODEL_DESCRIPTIONS = {
    "mlx-large-v3": "Quality-first full model; recommended for accuracy.",
    "mlx-large-v3-turbo": "Faster decoding with a smaller decoder.",
    "large-v3": "Quality-first full CPU model; higher memory demand.",
    "large-v3-turbo": "Faster full-size encoder with a smaller decoder.",
    "distil-large-v3.5": "Distilled English model for faster CPU recognition.",
    "distil-large-v3": "Earlier distilled English model for faster CPU recognition.",
    "medium.en": "Mid-sized English model; balances memory and accuracy.",
    "small.en": "Smaller English model; trades accuracy for speed.",
    "base.en": "Lightweight English model for low-memory use.",
    "tiny.en": "Smallest English model; prioritises speed over accuracy.",
}

# Apple silicon is preferred; CPU remains a selectable compatibility path.
MLX_CPU_FALLBACK = {
    "mlx-large-v3": "large-v3",
    "mlx-large-v3-turbo": "large-v3-turbo",
}


def apple_silicon() -> bool:
    import platform
    import sys
    return sys.platform == "darwin" and platform.machine() == "arm64"


def preferred_model() -> str:
    return "mlx-large-v3" if apple_silicon() else DEFAULT_MODEL
