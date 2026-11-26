# Developer-only speaker benchmark

No audio, model downloads or benchmark tooling are bundled with the app. Run on
an Apple silicon Mac using the diagnostic-enabled development build from PR #4.
Quit other NixScribe copies; open that build and leave it idle. The benchmark
uses its existing loopback API, with no token handling or audio uploads. Do not
operate the app UI during a run. Python 3.10+ and the already cached app models
are sufficient for the script itself.

## Public material

Use official VoxConverse (CC BY 4.0, copyright remains with original video
owners), https://mm.kaist.ac.kr/datasets/voxconverse/ . Cite J. S. Chung,
J. Huh, A. Nagrani, T. Afouras and A. Zisserman, "Spot the conversation:
speaker diarisation in the wild", Interspeech 2020. Annotations:
https://github.com/joonson/voxconverse (pin 24bf60be297701cd7e4ef18550c6d390c1b87365).
AMI is a second independent corpus for later evaluation:
https://groups.inf.ed.ac.uk/ami/download/ .

Download the official dev WAV archive from the dataset page and extract it
outside this repository. It is a substantial download; do not download the test
archive until the method is frozen. Verify the publisher's MD5 before extracting
(dev 2a6e07e7473d9841abb132554a698a36, test 834558bbd9b1ffd2d4893181556ceddd).
On macOS use md5 -q /path/to/archive.zip.

Fetch the annotations separately:

    git clone https://github.com/joonson/voxconverse.git ~/voxconverse-annotations
    git -C ~/voxconverse-annotations checkout 24bf60be297701cd7e4ef18550c6d390c1b87365

From the NixScribe checkout at the benchmark branch:

    python3 -m benchmarks.diarization \
      --audio-dir ~/Downloads/voxconverse-dev \
      --rttm-dir ~/voxconverse-annotations/dev \
      --split dev --output ~/NixScribe-benchmark-dev

Adjust the audio directory to the actual extraction location. The output must
not exist and should be outside the repository. The initial smoke sample is
the first matching ID (lexical order) in each of the 2, 3 and 4 speaker groups,
restricted to WAVs of 30-180 seconds. Use --per-group 3 for a larger dev sample.
The reference speaker count is not passed to the engines. The script saves a
manifest before inference, rotates mode order, and writes incremental scores
after each completed clip. If a job fails, keep the partial report and use a
new output directory for a rerun. Ctrl-C stops the client, not an active app job;
cancel that job in the app if necessary.

## Interpretation and promotion gate

* Word speaker accuracy: compare the exact same recognised words/timestamps
  across modes. Map anonymous labels one-to-one per recording to maximise
  matching reference words. Exclude missing timing, overlapping reference
  speakers, and words with <50% reference coverage. Unknown predictions count
  as errors. Report excluded counts. This is not word error rate or an
  independent word transcription reference.
* Raw DER: whole audio, no collar, overlap included; missed speaker time,
  false-alarm speaker time and speaker confusion; micro-average by reference
  speaker seconds. Not the official challenge scoring protocol. Combined v1
  emits word labels, not a continuous diarization timeline, so reporting DER
  for it would be misleading.
* Report aggregate and per-clip results, including stage times and approximate
  client wall time (polling overhead included). The runner rejects fallbacks,
  differing aligned words or model versions rather than silently comparing them.
* Develop any candidate acoustic rules on dev only; replay the stored
  diagnostics and original public WAVs without rerunning ASR. Freeze the rule,
  parameters, sample size and acceptance criterion before using the test split.
  Repeat the same command with test WAV/RTTM paths and --split test, using a
  new output directory. Require an improvement over the better baseline on
  held-out speaker-labelled words, check per-clip regressions and measure added
  time. A three-clip smoke test is not sufficient to claim generalisation.
  No claim about overlap recovery in combined output from this word metric.

The benchmark does not implement an acoustic candidate yet. Do not tune using
test data. Keep all audio, RTTM, diagnostics, reports and transcripts outside
Git; share aggregate numbers only. Do not redistribute the source recordings.

## Replay an existing development benchmark

No application, audio reads, inference, dependencies or network access are
needed after obtaining the source. From the repository root:

    python3 -m benchmarks.replay ~/NixScribe-benchmark-dev \
      --output ~/NixScribe-benchmark-dev/replay-report.json

Point the command at the completed result directory containing manifest.json,
report.json and the three per-recording job directories. It uses the original
RTTM paths from the manifest and the existing private diagnostic JSON files.
The original report must reproduce exactly for all word counts before any
candidate is scored. Output is a new aggregate/per-clip JSON with no words,
audio, speaker profiles or local paths; existing files are not replaced.

Both candidates anchor on the standalone MPS labels (not the combined run's
potentially different anonymous labels). MPS-first reverses the current
reconciliation priority while retaining the original MPS assignment if a
proposed label is unknown or unmatched. MPS-gap-only retains the standalone
assignment unless MPS has zero coverage and a single mapped Core ML speaker
covers at least half the word. Neither candidate introduces new speaker IDs.
The existing word score optimises the anonymous mapping separately per
candidate; the corrected/regressed audit additionally fixes the standalone
MPS mapping for both sides so changes are directly comparable. Replay timing
is CPU reconciliation time only, not a measurement of full inference speed.

These are development experiments, not a new app mode. No reference labels
enter the candidate rules; they are used only for evaluation. A tie with MPS
does not justify the cost of a second engine. Do not choose a winner from the
held-out test split; this tool deliberately rejects test manifests.


## Offline Community-1 clustering probe (development build)

Quit the app and run its bundled executable directly:

    /Applications/NixScribe2.app/Contents/MacOS/NixScribe --clustering-probe --preflight
    /Applications/NixScribe2.app/Contents/MacOS/NixScribe --clustering-probe \
      --audio /path/to/public-dev-recording.wav --output /path/to/new-output-directory

The probe performs only MPS diarization. It checks pinned package versions and
requires cached weights; Hub offline mode and disabled telemetry are set before
ML imports. No Whisper recognition, alignment, downloads or app API jobs.
The original upstream pipeline is run with a per-recording memo adapter for
segmentation and embeddings, with defensive copies on cache hits. The default
replay must match the capture before neighbouring speaker counts are evaluated.
The same recording is used for all trials. No reference labels are passed to
clustering. This is a research experiment, not an app feature. A separate local
scoring step must compare the resulting intervals against saved aligned words.
Share only the aggregate probe-report.json for public development clips. Keep
private-inference.npz and private-intervals.json outside the repository and
locally protected; they contain voice-derived data. No test-set tuning.

To verify a previous v1 capture across processes, use the same audio and model
snapshot with a new output directory:

    /path/to/NixScribe.app/Contents/MacOS/NixScribe --clustering-probe \
      --audio /path/to/recording.wav --replay-cache /path/to/old-probe \
      --output /path/to/new-replay

This loads the cached NPZ without pickle, verifies recorded identity, byte size,
shapes, window and numeric values, and replays every original setting through
upstream reconstruction. Both neural stage methods are fail-closed. Every
regular/exclusive interval must match the saved capture exactly. Model loading
and audio decoding still occur; no neural inference or ASR is run. Old v1
captures lack a stored cache checksum and centroids, so the new report records
the cache hash but cannot authenticate original array bytes or compare saved
centroids. Keep the original cache and intervals local. No parameter changes
until physical disk replay passes across the development set.
