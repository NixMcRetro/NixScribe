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

Do not advertise v5 in the application until an isolated production worker
reproduces the frozen reference-free selector and acoustic-short adjustment,
preserves manual speaker counts and single-speaker bypass, supports
cancellation and explicitly logged fallback, and passes cached development
parity and physical Apple silicon checks. Keep the existing Combined mode
distinct and labelled experimental. Avoid repeating neural inference.
