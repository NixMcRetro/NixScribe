# Audio alignment research: Phase A closeout (2026-11-27)

Status: held-out AMI acceptance gate PASS; original development gate remains unmet. Research only, no application integration or merge authorization.

- Frozen v4 SHA-256: `70fb889d3d13f40ca50a27420df14960cf346b22a9c2595b8f4de8e9358aa837`. v5 remains diagnostic only.
- Registered AMI protocol SHA-256: `280d8d2a77599cd067edfee4c330ebf10e7d84c8e397da1b02604758777e81ec`.
- Complete local report SHA-256: `78d67afcc3f297775c08abe5cb86dc5b7b1a7c1d61e23f660dbed4746407cec7`. Full report and frozen packages are preserved separately in the owner's research archive; no audio or raw diagnostics are published here.

| Metric | Development v4 | Held-out v4 |
| --- | ---: | ---: |
| Native cross-microphone | 9/10 | 9/10 |
| Positive offset/drift | 8/10 | 9/10 |
| Negative offset/drift | 8/10 | 9/10 |
| Same-mic control | 10/10 | 10/10 |
| Unrelated false accepts | 0/10 | 0/30 |
| Disjoint false accepts | not tested | 0/10 |

Held-out: 20/20 sources, 80/80 cases, 27/27 distinct v4 synthetic fixtures; median/p95/max accepted cross-mic nominal grid errors 3.55/18.37/23.47 ms. No incorrect accepted positive mapping or merge authorization. M5 arm64 local alignment calls total 23.00 s, p95 0.590 s, maximum 0.605 s; cumulative Python process peak RSS 758.48 MiB (includes acquisition and data; excludes child processes). Three correlated positive rejections occurred in one held-out meeting due to inadequate evidence across an interval. Different cohorts do not establish an algorithm improvement; ten meeting groups are not 80 independent observations, and nominal AMI synchronization is not independent acoustic or word-level ground truth.

## Next phase

Phase B: prospective local-only iPhone/QuickTime independent-clock validation. Begin with a short capture/measurement pilot, then separately register longer sessions (15, 30, 60 minutes), partial overlap, discontinuities and negative controls before testing. Measure against independently obtained acoustic landmarks, with uncertainty and missing evidence reported explicitly. Frozen v4 complementary feature fallback is limited to 15 minutes; longer runs must be reported as supported or rejected by the existing algorithm, not silently cropped or treated as equivalent. No tuning on the scored AMI holdout. No automatic merges; local interval confidence and word/speaker reconciliation are separate gates.

Issue #7 remains open. Preserve single-source behavior and all diarization modes, including Pyannote GPU + Auto default. No main changes, builds, release, version bump, model downloads, or external audio processing. This is a documentation-only checkpoint; it does not place the archived v4 code in the application repository.
