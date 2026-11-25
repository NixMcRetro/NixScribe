# NixScribe for macOS

A small local frontend for WhisperX for quality-first English transcription on Apple-silicon Macs.

## Features

- Eight selectable CPU Whisper models, with `large-v3` as the quality-first default. Model weights download on first use.
- English transcription with forced alignment for accurate timestamps.
- Pyannote Community-1 speaker diarization.
- Speaker selector: Auto or an exact 1-10 speakers.
- Real WhisperX stage progress and percentages.
- Live transcript view showing the most recently decoded timestamped text.
- Keeps the Mac awake only while a transcription is running.
- Creates a separate project folder for every transcription, named `YYYY-MM-DD Recording name` using the Mac's local date when the job starts. Repeated jobs on the same day receive `(2)`, `(3)` and so on; previous folders are never renamed or reused.
- Outputs `.txt`, `.srt`, and optional 128 kbps `.mp3` audio (enabled by default).
- Optional copy of the original recording into the project folder.
- Hugging Face token storage in macOS Keychain.
- Reuses the standard Hugging Face cache shared by compatible tools such as oMLX.

Example TXT output:

```text
[SPEAKER_00] Could you describe what happened next?

[SPEAKER_01] Yes. I started by checking the earlier recording.
```

## Recommended quality settings

For a recent Apple-silicon Mac with sufficient RAM, the default quality-first profile is:

- Model: `large-v3`
- Compute: `float32`
- Batch size: `4`
- Speaker count: choose the exact number when known

`int8` and `large-v3-turbo` remain available when speed matters more. Other model choices are `distil-large-v3.5`, `distil-large-v3`, `medium.en`, `small.en`, `base.en`, and `tiny.en`. These are Faster-Whisper CPU models, not GPU-accelerated engines. Smaller or distilled models can trade accuracy for speed. Selecting exactly one speaker skips pyannote diarization and does not require a Hugging Face token for that job.

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

## Experimental Apple GPU recognition (v0.3.0 development)

The model menu now groups the original eight CPU options separately from two
MLX Whisper options. The original CPU default is unchanged.

| GPU model | Download repository |
| --- | --- |
| large-v3 | mlx-community/whisper-large-v3-mlx |
| large-v3-turbo | mlx-community/whisper-large-v3-turbo |

MLX Whisper 0.4.3 and MLX/Metal 0.29.3 run the speech recognition pass on the
Apple GPU with FP16 weights. The first use downloads distinct MLX weights into
the standard Hugging Face cache. No cloud inference is performed. A separate
cancellable worker process releases the GPU model before CPU forced alignment
and speaker identification. TXT, SRT and MP3 outputs are unchanged.

The CPU engine's compute type, batch size, and hotwords are not applicable to
MLX. Context remains available. The MLX API does not provide the existing
WhisperX progress callback, so during recognition the app reports the active
stage without pretending to know its percentage. The transcript appears after
that pass. GPU errors are explicit; choose a CPU model to retry. The GPU and CPU
engines use different segmentation and decoding, so equivalence is not assumed.

The log includes ASR elapsed time and real-time factor (includes model loading,
excludes the separate model download).
Compare a repeat run with weights already cached on the same Mac and audio.
Use a trusted reference transcript to measure word error rate and inspect
speaker labels/timestamps. Hosted macOS CI validates packaging, not real GPU
performance. On physical Apple silicon, run:

```bash
/Applications/NixScribe.app/Contents/MacOS/NixScribe --self-test-mlx
```

This checks an actual GPU operation and MLX Whisper imports. Successful inference,
memory and accuracy benchmarks on physical hardware remain release gates.
Parakeet and Core ML evaluation are follow-up work, not available in this build.

## Model downloads

Before recognition, NixScribe checks the selected speech model's files in the
standard Hugging Face cache. A complete cached model is loaded directly without
a new download. If files are missing, the pinned status header says
**Downloading model**, followed by the model name. The app then shows
**Loading model** (CPU) or **GPU recognition** when it is ready. The detailed
log records the repository, cache hit or miss and completion. Downloads can be
cancelled and Hugging Face can reuse files already in its cache on the next run.

During downloads and other stages without a trustworthy percentage, the stage
indicator is indeterminate rather than displaying a fake estimate. The overall
job percentage stays at the completed processing milestone. Alignment and
diarization models may still perform their own first-use downloads later.

## Model publishers

The model menu shows the Hugging Face repository publisher and the exact
repository ID for the selected model. These are converted Whisper or
Distil-Whisper models, not eight different recognition engines.

| Model | Download repository |
| --- | --- |
| large-v3 | Systran/faster-whisper-large-v3 |
| large-v3-turbo | mobiuslabsgmbh/faster-whisper-large-v3-turbo |
| distil-large-v3.5 | distil-whisper/distil-large-v3.5-ct2 |
| distil-large-v3 | Systran/faster-distil-whisper-large-v3 |
| medium.en | Systran/faster-whisper-medium.en |
| small.en | Systran/faster-whisper-small.en |
| base.en | Systran/faster-whisper-base.en |
| tiny.en | Systran/faster-whisper-tiny.en |

The packaged backend self-test verifies the displayed repository IDs against
the pinned Faster-Whisper alias mappings without downloading model weights.

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
~/Documents/NixScribe
```

Every run gets its own folder. For example, transcribing `interview-01.m4a` produces:

```text
NixScribe/
└── interview-01/
    ├── interview-01.mp3
    ├── interview-01.txt
    └── interview-01.srt
```

If a project with the same recording name already exists, the next one is named `interview-01 (2)`, then `interview-01 (3)`, and so on. Existing projects are never silently overwritten.

**Save audio as MP3 (128 kbps)** is enabled by default. Audio is encoded from the original first audio stream, not the temporary transcription WAV; source metadata is not copied. **Keep a copy of the original media** is off by default. Enable it to retain the original alongside outputs; an original MP3 is never overwritten by the encoded copy. Old project folders are unchanged.

NixScribe creates a temporary 16 kHz mono PCM WAV inside the job’s hidden working directory before transcription. That working WAV is removed automatically after a successful job. If a job is cancelled or fails, the incomplete project folder and its working audio are removed. The original source media is never automatically deleted.

## Activity log

Expand Activity log and click **Copy Activity Log** to copy the full current
job as plain text to the macOS clipboard. The log no longer drops older
entries at an arbitrary line limit. It remains available after completion,
cancellation or failure until the next job starts or the app quits. The
viewport follows new entries only while at the bottom, and polling does not
replace selected text. Logs are not automatically uploaded or saved to disk;
check paths and technical details before sharing them.

## Live transcription view

While WhisperX is decoding, the app displays the most recently emitted timestamped segments and the approximate audio position being processed. Speaker labels are added after diarization, because the ASR stage does not yet know who is speaking. Exact speaker counts use WhisperX's `num_speakers` control, and unmatched segments are filled from the nearest diarized speaker turn rather than silently defaulting to speaker 0.

The final TXT and SRT are always written in chronological order even if internal batching processes chunks in a different order.

## GitHub Actions build

The repository includes `.github/workflows/build-macos.yml`. GitHub Actions builds the Apple-silicon app on the native `macos-26` ARM64 runner.

- Every push to `main` and pull request produces a downloadable development artifact retained for three days.
- Superseded development builds are cancelled automatically.
- A version tag matching the app version (starting with `v0.2.0`) triggers a permanent GitHub Release only after build and test success. Each published archive includes a SHA-256 checksum. Existing releases are not overwritten automatically.

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
~/Documents/NixScribe
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

## Native macOS window

NixScribe launches a native Cocoa/WebKit window instead of a separate browser
tab. Close the window, use **NixScribe > Quit NixScribe**, or click **Quit** to
terminate the application and its local Flask server. Closing is prevented while
transcription is underway. A second launch attempts to focus the existing window.

For developer/browser mode, run `python app.py --browser`.
The headless CI smoke test uses `WHISPERX_UI_NO_BROWSER=1`.

## TXT and SRT formatting

TXT now contains speaker-labelled paragraphs without timestamps, e.g.
`[SPEAKER_00] Hello.`. Consecutive segments from the same speaker are combined
by default; turn off that option if you want a paragraph for each segment.

SRT keeps timestamps. It normally includes speaker labels, a maximum of two
lines, 42 characters per line, and six seconds per cue. The export controls let
you toggle SRT speaker labels and change width, line count and cue duration.
Both exports and the final preview split recognition segments at actual
word-level speaker changes, rather than applying one dominant speaker to
every word. When word-level labels exist but a token has no matching label,
it is marked UNKNOWN instead of inventing an assignment. Original recognized
words, spelling and punctuation are retained. Valid aligned word times are
preserved; missing runs are interpolated between known boundaries.

SRT can join nearby same-speaker phrases across ASR segment boundaries.
Short cues prefer at least one second of reading time and roughly 20 spoken
characters per second, extending only into available silence (maximum 750 ms
of padding). Speaker boundaries, configured width/duration limits and the
audio end take precedence. Very fast dialogue can still produce short cues.

## Readable paragraphs, progress, and Dock icon

Long uninterrupted speaker turns are now separated into readable multi-sentence
paragraphs in TXT output without dropping text or timestamps from SRT.
A speaker label starts each TXT paragraph. A silence of at least 2.5 seconds
also breaks a paragraph.

The compact application header, including the name, version and tagline,
stays pinned at the top of the window while scrolling. A slim, full-width
progress strip appears immediately underneath when a job begins. It shows
overall progress, stage progress, elapsed time and the current activity,
without floating cards. A duplicate Start click no longer leaves a persistent
error banner: the button is disabled immediately while the first request is
pending, and transient messages clear when the job advances.

The supplied 1024 × 1024 RGBA artwork belongs at
`assets/NixScribe-master.png`. The build script `assets/make_icon.py` converts
it to a multi-resolution macOS ICNS without changing the design or the master
file. The ICNS is embedded in the signed `.app`. Until the master is committed,
development builds use the old blue waveform fallback. Stable release builds
require the supplied master artwork.

## Development roadmap

- v0.2.0: native application, expanded Faster-Whisper selection, MP3 export and new default project directory.
- v0.3.0: benchmark Apple silicon accelerated ASR backends against the existing CPU baseline, with timestamps and speaker labels preserved.
- v0.4.0: multi-source time alignment, independent transcripts, auditable reconciliation and explicit uncertain differences.
- v0.5.0: multi-model comparison and optional local language-model reconciliation, preserving the original transcript evidence. No cloud processing by default.

The v0.3.0 development branch implements experimental MLX Whisper selection; Parakeet, physical-device benchmarks, and the later multi-source stages remain planned.
