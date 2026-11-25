"""Shared version and supported CPU Whisper models."""
VERSION = "0.2.0"
MODELS = (
    ("large-v3", "Whisper large-v3 (quality-first)"),
    ("large-v3-turbo", "Whisper large-v3-turbo (faster)"),
    ("distil-large-v3.5", "Distil-Whisper large-v3.5 (English)"),
    ("distil-large-v3", "Distil-Whisper large-v3 (English)"),
    ("medium.en", "Whisper medium.en (English)"),
    ("small.en", "Whisper small.en (lightweight)"),
    ("base.en", "Whisper base.en (quick draft)"),
    ("tiny.en", "Whisper tiny.en (fastest draft)"),
)
DEFAULT_MODEL = "large-v3"
MODEL_IDS = frozenset(model for model, _ in MODELS)
