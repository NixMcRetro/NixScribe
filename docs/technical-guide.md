# NixScribe technical guide

[← Back to NixScribe](../README.md)

Detailed setup, model configuration, local storage, export behaviour and development notes for v0.3.0.

## Features

- MLX Whisper is preferred on Apple silicon, with eight selectable CPU Whisper models for compatibility and fallback. Model weights download on first use.
- English transcription with forced alignment for accurate timestamps.
- FluidAudio Core ML diarization, with Pyannote Community-1 MPS and CPU alternatives.
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

- Model: `large-v3 - MLX Community` (Apple GPU, FP16)
- Speaker identification: Core ML (FluidAudio, all compute units)
- Speaker count: choose the exact number when known
- CPU fallback uses the corresponding Whisper model in float32

`int8` and `large-v3-turbo` remain available when speed matters more. Other model choices are `distil-large-v3.5`, `distil-large-v3`, `medium.en`, `small.en`, `base.en`, and `tiny.en`. These are Faster-Whisper CPU models, not GPU-accelerated engines. Smaller or distilled models can trade accuracy for speed. Selecting exactly one speaker skips pyannote diarization and does not require a Hugging Face token for that job.

## Build from source

The included `build_macos_app.sh` creates an Apple-silicon application bundle:

```text
NixScribe.app
```

Python, WhisperX, Python dependencies and ffmpeg are bundled in the finished app. Model weights remain outside the app so they are not duplicated whenever the application is rebuilt.

The build must be performed on an Apple-silicon Mac because PyInstaller packages native macOS/arm64 dependencies on the machine performing the build.

### One-time build prerequisites

Install Xcode or a compatible Swift 6.2+ toolchain, then install the following Homebrew packages:

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

The build script applies an ad-hoc signature suitable for a locally built personal application. Developer ID signing and notarisation are separate distribution steps. macOS may block downloaded, unnotarised builds; only approve an app you have verified and trust. See Apple's [guidance on opening apps safely](https://support.apple.com/en-au/102445), or build from source.

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

## Apple GPU recognition (MLX)

The model menu now groups the original eight CPU options separately from two
MLX Whisper options. MLX large-v3 is the default on Apple silicon.

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
that pass. A failed MLX download or inference automatically retries the matching CPU model (large-v3 or large-v3-turbo) in float32. Cancellation never starts a fallback. Both attempted and successful models are logged. The GPU and CPU
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

This checks an actual GPU operation and MLX Whisper imports. Initial short-recording MLX, MPS and Core ML inference succeeded on a physical M5 Mac. Longer recordings, memory use, fallback behaviour and accuracy still need testing. Parakeet remains optional follow-up work.

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
└── 2026-11-25 interview-01/
    ├── interview-01.mp3
    ├── interview-01.txt
    └── interview-01.srt
```

If a project with the same dated recording name already exists, the next one is named `2026-11-25 interview-01 (2)`, then `2026-11-25 interview-01 (3)`, and so on. Existing projects are never silently overwritten.

**Save audio as MP3 (128 kbps)** is enabled by default. Audio is encoded from the original first audio stream, not the temporary transcription WAV; source metadata is not copied. **Keep a copy of the original media** is off by default. Enable it to retain the original alongside outputs; an original MP3 is never overwritten by the encoded copy. Old project folders are unchanged.

NixScribe creates a temporary 16 kHz mono PCM WAV inside the job’s hidden working directory before transcription. That working WAV is removed automatically after a successful job. If a job is cancelled or fails, the incomplete project folder and its working audio are removed. The original source media is never automatically deleted.

## Apple GPU speaker identification (Pyannote MPS)

Core ML (FluidAudio) is preferred on Apple silicon; MPS and CPU remain selectable. A failed Core ML attempt retries MPS, then CPU. Explicit CPU selection remains CPU-only.
Choose the speaker identification backend independently of speech recognition.
The MPS and CPU options use the same Pyannote Community-1 model and Hugging Face cache.
Single-speaker mode skips diarization altogether. Word alignment remains CPU.

The experimental option uses PyTorch MPS in a separate process. A native
crash, unavailable MPS device, unsupported operation, invalid output or empty
speaker intervals when speech was recognized causes an explicitly logged CPU
retry. Unsupported MPS operations are not silently redirected by PyTorch's
fallback environment variable. Some pipeline work, such as clustering, can
still run on CPU. The log records requested and successful device and elapsed
time (including model loading; separate from any failed attempt).

Compare CPU and GPU on identical local media and speaker counts; speed is not
a guarantee of equal diarization accuracy. Speaker IDs may swap between runs.
No inference or benchmarks on physical MPS hardware have been verified by CI.
The bundled application can check actual MPS arithmetic via:

```bash
/Applications/NixScribe.app/Contents/MacOS/NixScribe --self-test-mps
```

A passing arithmetic test does not establish successful model inference.
### Core ML (FluidAudio)

The preferred option uses a bundled native Swift helper with FluidAudio v0.18.4
(Apache-2.0, https://github.com/FluidInference/FluidAudio), pinned to that
release with only its Diarizer trait enabled. Core ML uses all compute units
(.all), allowing CPU, GPU and ANE scheduling; the Neural Engine is not
guaranteed to execute every operation. First use can download separate
model weights under ~/Library/Application Support/FluidAudio/Models.
The models are checked offline first; audio inference is local.
The existing Hugging Face token is still required for multi-speaker jobs
because the fallback uses the gated Pyannote model. Exact speaker count
is passed as a target, not a guarantee. On failure the app logs the reason
and retries MPS, then CPU diarization. Source builds require Swift 6.2 or later
(macOS 15+); build the helper with
`swift build --package-path native-diarizer -c release`.
Initial short-recording tests succeeded on an M5 Mac; longer-recording accuracy and memory benchmarks remain pending.
The build includes FluidAudio's Apache-2.0 license.

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

## Run the development app from source

```bash
./setup.sh
swift build --package-path native-diarizer -c release
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

NixScribe accepts common audio and video containers such as WAV, M4A, MP3, FLAC, AAC, MKV, MP4, MOV, WebM and AVI. For every job it extracts the first audio stream to a temporary 16 kHz mono PCM WAV, uses that normalized WAV for WhisperX transcription and speaker diarization, then removes the temporary working audio when the job finishes. For Core ML in source mode, build the native helper first using the Swift command above. The original source file is never modified or deleted.

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

v0.3.0 establishes MLX and Core ML as the Apple silicon defaults with automatic fallback. [Issue #3](https://github.com/NixMcRetro/NixScribe/issues/3) tracks v0.3.1 combined diarization and accuracy improvements. Parakeet evaluation and the later multi-source stages remain optional future work.
