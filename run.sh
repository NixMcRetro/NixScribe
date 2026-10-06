#!/bin/zsh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment not found. Run ./setup.sh first."
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" ]]; then
  echo "No HF_TOKEN environment variable is set."
  echo "On macOS you can save a Hugging Face token in NixScribe's Keychain settings instead."
  echo "Or export HF_TOKEN='hf_...' before launching."
  echo
fi

export WHISPERX_WORKDIR="${WHISPERX_WORKDIR:-$HOME/Music}"
export WHISPERX_PROJECTS_DIR="${WHISPERX_PROJECTS_DIR:-$HOME/Documents/NixScribe Transcriptions}"
exec .venv/bin/python app.py
