"""Conservative, local comparison of two diarization engines.

Intervals use the same validated transport as the existing workers.
Overlap is timing evidence, not calibrated confidence or ground truth.
"""
from bisect import bisect_right
import math

from diarization_backend import validate_intervals


def _tracks(rows):
    tracks = {}
    for row in validate_intervals(rows):
        spans = tracks.setdefault(row["speaker"], [])
        start, end = row["start"], row["end"]
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
        else:
            spans.append((start, end))
    return tracks


def _intersection(left, right):
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0.0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def _unique_best(scores):
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    if not ranked or scores[ranked[0]] <= 0:
        return None
    if len(ranked) > 1 and math.isclose(scores[ranked[0]], scores[ranked[1]],
                                        rel_tol=1e-9, abs_tol=1e-9):
        return None
    return ranked[0]


def _mapping(core, mps):
    overlaps = {b: {a: _intersection(core[a], mps[b]) for a in sorted(core)}
                for b in sorted(mps)}
    mapping = {}
    used = set(core) | {"UNKNOWN"}
    for b, scores in overlaps.items():
        a = _unique_best(scores)
        # A reciprocal unique match avoids collapsing two speakers into one.
        if a is not None and _unique_best({k: v[a] for k, v in overlaps.items()}) == b:
            mapping[b] = a
        else:
            index = 0
            while f"SPEAKER_{index:02d}" in used:
                index += 1
            mapping[b] = f"SPEAKER_{index:02d}"
            used.add(mapping[b])
    return mapping, overlaps


def _index(tracks):
    return {label: (spans, [end for _, end in spans])
            for label, spans in tracks.items()}


def _coverage(index, start, end):
    scores = {}
    for label, (spans, ends) in index.items():
        total = 0.0
        for i in range(bisect_right(ends, start), len(spans)):
            a, b = spans[i]
            if a >= end:
                break
            total += min(b, end) - max(a, start)
        if total > 0:
            scores[label] = total
    return scores


def reconcile(coreml, mps, words, check_cancel=lambda: None):
    """Return mapping, overlap evidence and a decision per aligned word.

    Prefer clear agreements; use the only unique covered speaker if the other
    engine has no coverage. On conflicting unique assignments retain Core ML
    unless MPS fully covers the word and Core ML covers less than half.
    Unresolved ties are explicit, not inferred from neighbouring speech.
    """
    check_cancel()
    core, secondary = _tracks(coreml), _tracks(mps)
    mapping, overlaps = _mapping(core, secondary)
    primary_index = _index(core)
    secondary_index = _index({mapping[k]: v for k, v in secondary.items()})
    decisions = []
    for word in words:
        check_cancel()
        try:
            start, end = float(word["start"]), float(word["end"])
            valid = math.isfinite(start) and math.isfinite(end) and 0 <= start < end
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            decisions.append({"speaker": "UNKNOWN", "reason": "missing_timing",
                              "coreml_overlap": {}, "mps_overlap": {}, "uncertain": True})
            continue
        a, b = _coverage(primary_index, start, end), _coverage(secondary_index, start, end)
        first, second = _unique_best(a), _unique_best(b)
        uncertain = True
        if first is not None and first == second:
            speaker, reason = first, "agreement"
            uncertain = len(a) != 1 or len(b) != 1
        elif first is not None and not b:
            speaker, reason = first, "coreml_only_coverage"
        elif second is not None and not a:
            speaker, reason = second, "mps_only_coverage"
        elif first is not None and second is not None:
            if b[second] >= (end - start) - 1e-9 and a[first] < (end - start) / 2:
                speaker, reason = second, "mps_full_coreml_partial"
            else:
                speaker, reason = first, "coreml_disagreement"
        elif first is not None:
            speaker, reason = first, "coreml_mps_ambiguous"
        elif second is not None:
            speaker, reason = second, "mps_coreml_ambiguous"
        else:
            speaker, reason = "UNKNOWN", "ambiguous_or_uncovered"
        decisions.append({"speaker": speaker, "reason": reason,
                          "coreml_overlap": a, "mps_overlap": b, "uncertain": uncertain})
    return {"mapping": mapping, "mapping_overlap_seconds": overlaps, "decisions": decisions}
