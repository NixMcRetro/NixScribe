# NixScribe for macOS

A small local frontend for WhisperX for quality-first English transcription on Apple-silicon Macs.

## Features

- Whisper `large-v3` by default, with `large-v3-turbo` available for faster jobs.
- English transcription with forced alignment for accurate timestamps.
- Pyannote Community-1 speaker diarization.
- Speaker selector: Auto or an exact 1-10 speakers.
- Real WhisperX stage progress and percentages.
- Live transcript view showing the most recently decoded timestamped text.
- Keeps the Mac awake only while a transcription is running.
- Creates a separate project folder for every transcription.
- Outputs only `.txt` and `.srt`.
- Optional copy of the original recording into the project folder.
- Hugging Face token storage in macOS Keychain.
- Reuses the standard Hugging Face cache shared by compatible tools such as oMLX.

Example TXT output:

```text
[00:03:12.440 --> 00:03:18.020] SPEAKER_00: Could you describe what happened next?
[00:03:18.310 --> 00:03:21.770] SPEAKER_01: Yes. I started by checking the earlier recording.
```

## Recommended quality settings

For a recent Apple-silicon Mac with sufficient RAM, the default quality-first profile is:

- Model: `large-v3`
- Compute: `float32`
- Batch size: `4`
- Interleaved context: enabled
- Speaker count: choose the exact number when known

`int8` and `large-v3-turbo` remain available when speed matters more.

## Build a self-contained macOS app

The included `build_macos_app.sh` creates an Apple-silicon application bundle:

```text
NixScribe.app
```

Python, WhisperX, Python dependencies and ffmpeg are bundled in the finished app. Model weights remain outside the app so they are not duplicated whenever the application is rebuilt.

The build must be performed on an Apple-silicon Mac because PyInstaller packages native macOS/arm64 dependencies on the machine performing the build.

### One-time build prerequisites

```bash
brew install python@3.13 ffmpeg
```

Then, from the project folder:

```bash
chmod +x build_macos_app.sh
./build_macos_app.sh
```

The results are placed in `dist/`:

```text
dist/NixScribe.app
dist/NixScribe-macOS-arm64.zip
```

Drag `NixScribe.app` to `/Applications` and launch it from Finder or Spotlight.

The build script applies an ad-hoc signature suitable for a locally built personal application. Developer ID signing and notarization are separate distribution steps if you want to publish prebuilt binaries for other Macs.

## First launch and speaker diarization

Speaker diarization uses:

```text
pyannote/speaker-diarization-community-1
```

Accept the model terms on Hugging Face, then paste a Hugging Face read token into the app and click **Save token**. The token is stored in macOS Keychain and is never returned to the browser UI.

When running from source, `HF_TOKEN` is also supported:

```bash
export HF_TOKEN='hf_...'
```

## Model cache and oMLX compatibility

The app deliberately does **not** create a private model cache. It follows the standard Hugging Face Hub cache resolution used by `huggingface_hub`:

1. `HF_HUB_CACHE`, when set.
2. `$HF_HOME/hub`, when `HF_HOME` is set.
3. Otherwise `~/.cache/huggingface/hub`.

That means compatible model downloads can be shared with other Hugging Face clients. Current oMLX also understands this standard Hugging Face cache layout, so no separate "WhisperX model cache" is normally needed.

The oMLX model directory (normally `~/.omlx/models`, or another configured model directory) is different from the Hugging Face cache. It generally contains MLX-format LLM/VLM models. WhisperX does not scan that directory because those model formats are not interchangeable with Faster-Whisper/CTranslate2, PyTorch alignment models, or Pyannote diarization models.

PyTorch-specific assets use the normal PyTorch cache (`TORCH_HOME` when configured, otherwise the platform/default cache such as `~/.cache/torch`).

## Transcription projects

Choose a **Transcription projects location** in the app. By default the app proposes:

```text
~/Documents/NixScribe Transcriptions
```

Every run gets its own folder. For example, transcribing `interview-01.m4a` produces:

```text
NixScribe Transcriptions/
└── interview-01/
    ├── interview-01.m4a
    ├── interview-01.txt
    └── interview-01.srt
```

If a project with the same recording name already exists, the next one is named `interview-01 (2)`, then `interview-01 (3)`, and so on. Existing projects are never silently overwritten.

**Copy the original recording into each project folder** is enabled by default. Disable it if you do not want to duplicate large recordings; the source is then read in place and is never modified or deleted.

NixScribe creates a temporary 16 kHz mono PCM WAV inside the job’s hidden working directory before transcription. That working WAV is removed automatically after a successful job. If a job is cancelled or fails, the incomplete project folder and its working audio are removed. The original source media is never automatically deleted.

## Live transcription view

While WhisperX is decoding, the app displays the most recently emitted timestamped segments and the approximate audio position being processed. Speaker labels are added after diarization, because the ASR stage does not yet know who is speaking.

The final TXT and SRT are always written in chronological order even if internal batching processes chunks in a different order.

## GitHub Actions build

The repository includes `.github/workflows/build-macos.yml`. GitHub Actions builds the Apple-silicon app on the native `macos-26` ARM64 runner.

- Every push to `main` produces a downloadable Actions artifact.
- Pull requests run the same build as a packaging check.
- A tag such as `v0.1.0` also creates or updates a GitHub Release and attaches `NixScribe-macOS-arm64.zip`.

The workflow does not download Whisper model weights. Models are resolved on the Mac that runs NixScribe and remain in the normal Hugging Face/PyTorch caches.

## Run from source

```bash
./setup.sh
./run.sh
```

By default, the source-mode file browser starts around:

```text
~/Music
```

and transcription projects default to:

```text
~/Documents/NixScribe Transcriptions
```

Both can be overridden:

```bash
export WHISPERX_WORKDIR="$HOME/Music/Recordings"
export WHISPERX_PROJECTS_DIR="$HOME/Documents/Transcripts"
./run.sh
```

NixScribe accepts common audio and video containers such as WAV, M4A, MP3, FLAC, AAC, MKV, MP4, MOV, WebM and AVI. For every job it extracts the first audio stream to a temporary 16 kHz mono PCM WAV, uses that normalized WAV for WhisperX transcription and speaker diarization, then removes the temporary working audio when the job finishes. The original source file is never modified or deleted.

## GitHub hygiene

The included `.gitignore` excludes virtual environments, build output, local caches, tokens, recordings, and transcription projects. Keep private audio/transcripts and Hugging Face credentials out of the repository.
