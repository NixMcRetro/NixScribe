"""Local prompt preparation shared by preview, validation and recognition."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from app_config import MODEL_IDS, MLX_REPOSITORIES, MODEL_REPOSITORIES

# MLX retains the last 223 prompt tokens. Faster-Whisper caps each field
# at 223 separately; our conservative combined CPU budget preserves space
# for output. Reject rather than silently truncate either input.
PROMPT_LIMIT = 223
MAX_FIELD_CHARS = 16000


class PromptError(ValueError):
    pass


def prompt_fields(payload):
    fields = tuple(payload.get(key, "") for key in ("initial_prompt", "hotwords"))
    if any(not isinstance(v, str) or len(v) > MAX_FIELD_CHARS for v in fields):
        raise PromptError("Prompt fields must be text, at most 16,000 characters each.")
    return tuple(v.strip() for v in fields)


def _mlx_encoder():
    from mlx_whisper.tokenizer import get_tokenizer
    return get_tokenizer(True, num_languages=100, language="en", task="transcribe").encode


def _cpu_encoder(model, snapshot):
    from tokenizers import Tokenizer
    if snapshot is None:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(MODEL_REPOSITORIES[model], "tokenizer.json")
        if not isinstance(cached, str):
            return None
        path = Path(cached)
    else:
        path = Path(snapshot) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def prepare_prompt(model, initial_prompt, hotwords, *, snapshot=None):
    if model not in MODEL_IDS and model not in MLX_REPOSITORIES:
        raise PromptError("Unsupported recognition model.")
    initial_prompt, hotwords = prompt_fields(
        {"initial_prompt": initial_prompt, "hotwords": hotwords})
    mlx = model in MLX_REPOSITORIES
    effective = "\n\n".join(v for v in (initial_prompt, hotwords) if v) if mlx else initial_prompt
    result = {
        "engine": "mlx" if mlx else "cpu",
        "initial_prompt": effective, "hotwords": "" if mlx else hotwords,
        "initial_tokens": 0, "vocabulary_tokens": 0, "effective_tokens": 0,
        "limit": PROMPT_LIMIT, "available": True, "over_limit": False,
    }
    if not (initial_prompt or hotwords):
        return result
    try:
        encode = _mlx_encoder() if mlx else _cpu_encoder(model, snapshot)
        if encode is None:
            result.update(available=False, initial_tokens=None,
                          vocabulary_tokens=None, effective_tokens=None)
            return result
        def count(text):
            return len(encode(" " + text)) if text else 0
        initial_count, vocab_count = count(initial_prompt), count(hotwords)
        total = count(effective) if mlx else initial_count + vocab_count
    except Exception:
        # Tokenizer errors may include the input text; never expose that text.
        raise PromptError("Could not count prompt tokens with the local tokenizer.") from None
    result.update(initial_tokens=initial_count, vocabulary_tokens=vocab_count,
                  effective_tokens=total, over_limit=total > PROMPT_LIMIT)
    return result


def require_prompt(plan):
    if not plan["available"]:
        raise PromptError("Local tokenizer is unavailable; prompt cannot be validated.")
    if plan["over_limit"]:
        raise PromptError(f"Prompt uses {plan['effective_tokens']} tokens; shorten to "
                          f"{plan['limit']} or fewer. No vocabulary was discarded.")
    return plan


def prompt_metadata(plan):
    return {key: plan[key] for key in ("engine", "initial_tokens", "vocabulary_tokens",
                                      "effective_tokens", "limit")}

# Only explicit profile save/load actions persist user text.
PROFILE_MAX_BYTES = 256 * 1024


def _profile_data(payload):
    if not isinstance(payload, dict):
        raise PromptError("Invalid vocabulary profile.")
    prompt_fields(payload)
    return {"schema": 1, "initial_prompt": payload.get("initial_prompt", ""),
            "hotwords": payload.get("hotwords", "")}


def save_profile(path, payload):
    data = _profile_data(payload)
    path = Path(path)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, prefix=".nixscribe-profile-",
                                         delete=False) as f:
            temp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def load_profile(path):
    path = Path(path)
    if not path.is_file():
        raise PromptError("Choose a regular vocabulary profile file.")
    with path.open("rb") as f:
        raw = f.read(PROFILE_MAX_BYTES + 1)
    if len(raw) > PROFILE_MAX_BYTES:
        raise PromptError("Vocabulary profile is too large.")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise PromptError("Invalid vocabulary profile JSON.") from None
    if (not isinstance(data, dict) or set(data) != {"schema", "initial_prompt", "hotwords"}
            or type(data["schema"]) is not int or data["schema"] != 1):
        raise PromptError("Unsupported vocabulary profile format.")
    return _profile_data(data)
