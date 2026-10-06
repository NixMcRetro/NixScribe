#!/bin/zsh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found. Install it first:"
  echo "  brew install ffmpeg"
  exit 1
fi

PY=""
for candidate in python3.13 /opt/homebrew/bin/python3.13 /usr/local/bin/python3.13; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PY="$(command -v "$candidate")"
    break
  fi
done

if [[ -z "$PY" ]]; then
  echo "Python 3.13 is required for current WhisperX 3.8.x."
  echo "Install it with:"
  echo "  brew install python@3.13"
  exit 1
fi

"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "Setup complete."
echo "Before first diarization, accept access for:"
echo "  https://huggingface.co/pyannote/speaker-diarization-community-1"
echo
echo "Then run ./run.sh and save your Hugging Face token in NixScribe's settings,"
echo "or export HF_TOKEN='hf_...' before launching."
