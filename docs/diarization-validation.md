# v0.3.1 diarization validation

Frozen independent VoxConverse test cohort: 30 recordings, 10 each with two,
three and four reference speakers, 90.84 minutes, 13,432 evaluated aligned
recognised words (1,327 excluded). Speaker attribution is not transcription
word accuracy. Annotation commit: `24bf60be297701cd7e4ef18550c6d390c1b87365`.

| Method | Development accuracy (15 clips) | Held-out accuracy (30 clips) | Held-out raw DER | Count correct |
| --- | ---: | ---: | ---: | ---: |
| Pyannote automatic | 98.198% | 96.091% | 13.979% | 23/30 |
| Intelligent Auto v4 | 98.812% | 96.158% | 13.969% | 22/30 |
| Intelligent Auto v5 | 98.915% | 96.166% | 13.969% | 22/30 |

v5 gained ten net correct words, but lost one correct speaker count.
The paired recording-level 95% bootstrap interval for improvement was
-0.015 to +0.183 percentage points. This does not establish a reliable
improvement; keep original Pyannote as default. Both methods scored 99.717%
on the two-speaker subset. The existing Combined experimental mode is a
different algorithm, not v5.

The private research archive preserves frozen source, manifests and numerical
reports. No audio, transcripts, embeddings or personal paths are committed.
Do not tune on this now-unblinded cohort and describe it as independent.

## Experimental v5 integration gate

The draft app integrates the frozen reference-free selector and acoustic-short
adjustment as **Intelligent Auto v5 (Experimental)**. It is available only
with automatic speaker count on Apple silicon. Its isolated worker captures
Pyannote segmentation and embeddings once, then moves to CPU for five cached
candidate reconstructions. The separately computed FluidAudio intervals are
used only for acoustic-short refinement. If the experimental replay fails,
the captured standard Pyannote intervals are returned and the normal word
assignment path is used; if the worker fails outright, CPU Pyannote is the
fallback. A missing Core ML result uses original Pyannote.

The worker checks the exact research dependency versions and Community-1
snapshot/configuration before enabling the experimental path. Different
versions safely fall back; the dependency set is unchanged. The existing
Combined mode remains distinct and experimental. No private inference cache
is persisted by the production worker. The source algorithm functions were
extracted from the verified archive (SHA-256
`b7ab64c45ed4c0e26344490bec190ea9c2ea656b3b073df5051425202a5bb029`).

**Release gate:** synthetic unit tests do not establish real-audio production
parity. Verify the production cache-only implementation against all 15
development clips and the frozen independent predictions without tuning,
then perform physical Apple silicon checks of the packaged worker, fallback,
cancellation, single/manual speaker count and exports. Do not claim a
generalised accuracy improvement.
