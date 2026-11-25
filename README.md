<p align="center">
  <img src="assets/NixScribe-master.png" alt="NixScribe app icon" width="140">
</p>

<h1 align="center">NixScribe</h1>

<p align="center"><strong>Your recordings. Your Mac. Your words.</strong></p>

<p align="center">
  <img alt="Version 0.3.0" src="https://img.shields.io/badge/version-0.3.0-7855a6">
  <img alt="Apple silicon" src="https://img.shields.io/badge/macOS-Apple%20silicon-555555">
  <img alt="Local transcription" src="https://img.shields.io/badge/transcription-local-2c875f">
  <a href="https://github.com/NixMcRetro/NixScribe/actions/workflows/build-macos.yml"><img alt="macOS build status" src="https://github.com/NixMcRetro/NixScribe/actions/workflows/build-macos.yml/badge.svg?branch=main"></a>
</p>

NixScribe turns audio and video recordings into readable, speaker-labelled English transcripts and timed subtitles. It runs locally on Apple silicon, putting your Mac's GPU and Neural Engine to work without sending recordings to a transcription service.

## ✨ What it does

- 🎙️ **Transcribe** interviews, conversations, meetings and videos.
- 👥 **Identify speakers** automatically or choose the expected number.
- ⚡ **Use Apple silicon** with MLX Whisper and FluidAudio Core ML, plus automatic fallbacks.
- 📝 **Export your words** as plain text, SRT subtitles and optional MP3 audio.
- 🔒 **Keep it local** with reusable model downloads and Keychain token storage.
- 📁 **Stay organised** with a fresh, dated project folder for every job.

## 🚀 Get started

**Requirements:** an Apple silicon Mac running macOS 15 or later, enough free memory and disk space for the chosen models, and internet access for the initial model downloads.

1. Download the app from [GitHub Releases](https://github.com/NixMcRetro/NixScribe/releases) when a tagged release is available, or use the [latest successful macOS build](https://github.com/NixMcRetro/NixScribe/actions/workflows/build-macos.yml?query=branch%3Amain+is%3Asuccess). Development builds are listed under **Artifacts** and expire after three days.
2. Extract the archive, move `NixScribe.app` to Applications and open it. The current build is ad-hoc signed, not notarised; see the [setup guide](docs/technical-guide.md#build-from-source) if macOS blocks it.
3. Choose a recording, set the expected number of speakers and press **Start**.

For multiple speakers, first accept the [Pyannote Community-1 model terms](https://huggingface.co/pyannote/speaker-diarization-community-1), create a [Hugging Face read token](https://huggingface.co/settings/tokens) and save it in NixScribe. This enables the fallback engines. A single-speaker job skips diarization and doesn't require that token.

The first transcription may download several models. Later jobs reuse the cached weights.

## 🍎 Built for Apple silicon

| Stage | Preferred engine | Automatic fallback |
| --- | --- | --- |
| Speech recognition | MLX Whisper large-v3 (GPU) | Faster-Whisper (CPU) |
| Word alignment | WhisperX (CPU) | - |
| Speaker identification | FluidAudio Core ML | Pyannote MPS, then CPU |

Core ML can use the CPU, GPU and Neural Engine where supported. Apple decides which hardware executes each operation. All recognition and speaker identification engines remain manually selectable.

### A little speed goes a long way

An initial test on an M5 Mac with an 83.1-second recording and cached models:

| Configuration | Total time |
| --- | ---: |
| CPU recognition + CPU diarization | 78 s |
| CPU recognition + MPS diarization | 58 s |
| MLX recognition + MPS diarization | 32 s |
| MLX recognition + Core ML diarization | **14 s** |

That's approximately **5.6× faster** than the original CPU pipeline on this sample. Core ML speaker identification took just 0.6 seconds.

These are individual development test runs, not guaranteed performance or accuracy figures. Speaker attribution can differ between engines; longer recordings and accuracy tuning are planned.

## 📂 A place for everything

Every transcription gets its own project under `~/Documents/NixScribe` by default:

```text
NixScribe/
└── 2026-11-25 Interview/
    ├── Interview.txt
    ├── Interview.srt
    └── Interview.mp3
```

The TXT file contains readable paragraphs without timestamps:

```text
[SPEAKER_00] Could you describe what happened next?

[SPEAKER_01] Yes. I started by checking the earlier recording.
```

SRT retains timestamps with configurable subtitle formatting. MP3 export is optional and enabled by default. You can also keep a copy of the original recording in the project.

**Your original media stays untouched.** Repeated jobs get numbered folders rather than overwriting previous results.

## 🛠️ Under the hood

Want to build it yourself, compare engines or understand the model cache? The [technical guide](docs/technical-guide.md) covers:

- Building the self-contained macOS app and running from source.
- All available models, repository IDs and hardware settings.
- Model downloads, cache locations and oMLX compatibility.
- Speaker identification, formatting, activity logs and fallback behaviour.
- GitHub Actions, self-tests and release packaging.

NixScribe is powered by [WhisperX](https://github.com/m-bain/whisperX), [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [FluidAudio](https://github.com/FluidInference/FluidAudio), [Pyannote](https://github.com/pyannote/pyannote-audio) and [FFmpeg](https://ffmpeg.org/).

## 🧭 What's next?

| Version | Focus |
| --- | --- |
| v0.2.0 | Native macOS app and audio export |
| **v0.3.0** | **Apple silicon acceleration** |
| [v0.3.1](https://github.com/NixMcRetro/NixScribe/issues/3) | Combined diarization and accuracy improvements |
| v0.4.0 | Multi-source transcription and synchronisation |
| v0.5.0 | Multi-model comparison and optional local AI-assisted reconciliation |

*Less time waiting. More time listening.*
