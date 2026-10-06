#!/bin/zsh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This app must be built on macOS."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This build is configured for Apple silicon (arm64)."
  exit 1
fi

PY=""
for candidate in /opt/homebrew/bin/python3.13 python3.13; do
  if [[ -x "$candidate" ]]; then
    PY="$candidate"
    break
  elif command -v "$candidate" >/dev/null 2>&1; then
    PY="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$PY" ]]; then
  echo "Python 3.13 is needed to build the app."
  echo "Install it once with: brew install python@3.13"
  exit 1
fi

FFMPEG="$(command -v ffmpeg 2>/dev/null || true)"
FFPROBE="$(command -v ffprobe 2>/dev/null || true)"
if [[ -z "$FFMPEG" ]]; then
  echo "ffmpeg is needed at build time so it can be bundled into the app."
  echo "Install it once with: brew install ffmpeg"
  exit 1
fi

BUILD_VENV="$HERE/.build-venv"
rm -rf "$BUILD_VENV" "$HERE/build" "$HERE/dist"
"$PY" -m venv "$BUILD_VENV"
source "$BUILD_VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-build.txt

export WHISPERX_BUNDLED_FFMPEG="$FFMPEG"
export WHISPERX_BUNDLED_FFPROBE="$FFPROBE"

python -m PyInstaller --noconfirm --clean NixScribe.spec

APP="$HERE/dist/NixScribe.app"
if [[ ! -d "$APP" ]]; then
  echo "Build did not produce the expected app bundle."
  exit 1
fi

# Re-apply an ad-hoc signature after all nested binaries are in place. This is
# sufficient for a locally built personal app; Developer ID notarization is a
# separate step only needed for wider distribution.
/usr/bin/codesign --force --deep --sign - "$APP"

ZIP="$HERE/dist/NixScribe-macOS-arm64.zip"
rm -f "$ZIP"
/usr/bin/ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"

cat <<DONE

Build complete.

App:
  $APP

Portable archive:
  $ZIP

You can drag the .app to /Applications. Python, WhisperX and ffmpeg are inside
it; Homebrew is not required to run the finished app.

The first use of large-v3 / alignment / diarization will still download their
model weights into your normal Hugging Face/PyTorch caches. Those weights are
kept outside the .app so rebuilding the program does not duplicate many GB.
DONE
