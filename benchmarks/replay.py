"""Replay completed development benchmarks without audio, inference or network."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from benchmarks.diarization import MODES, best_mapping, compare, read_rttm, word_reference, word_score
from speaker_reconciliation import reconcile

CANDIDATES = ("mps_first", "mps_gap_only")
SCORE_FIELDS = ("correct", "evaluated", "excluded", "unknown")


def candidates(diagnostics):
    core, mps = diagnostics["coreml"], diagnostics["mps"]
    words = mps["aligned_words"]
    baseline = mps["assigned_speakers"]
    raw_mps = mps["raw_intervals"]["mps"]
    known = {row["speaker"] for row in raw_mps} - {"UNKNOWN"}
    if len(baseline) != len(words) or not set(baseline) <= known | {"UNKNOWN"}:
        raise ValueError("Standalone MPS assignments do not match its raw speaker labels")

    # Swap engine roles for the experiment only. The existing function names
    # its primary coverage coreml_overlap and its secondary coverage mps_overlap.
    replay = reconcile(raw_mps, core["raw_intervals"]["coreml"], words)
    first, gap = [], []
    for word, original, decision in zip(words, baseline, replay["decisions"]):
        label = decision["speaker"]
        primary = decision["coreml_overlap"]
        secondary = decision["mps_overlap"]
        # Never invent a new speaker or replace a known label with UNKNOWN.
        first.append(label if label in known else original)
        if (not primary and len(secondary) == 1 and label in known
                and secondary.get(label, 0) > 0
                and secondary[label] >= (word["end"] - word["start"]) / 2):
            gap.append(label)
        else:
            gap.append(original)
    return dict(mps_first=first, mps_gap_only=gap)


def audit(labels, baseline, candidate):
    """Use the standalone MPS mapping for both sides of the change audit."""
    mapping = best_mapping(Counter((h, r) for h, r in zip(baseline, labels)
                                   if h != "UNKNOWN" and r is not None))
    changes = corrected = regressed = evaluated_changes = 0
    for truth, before, after in zip(labels, baseline, candidate):
        if before == after:
            continue
        changes += 1
        if truth is None:
            continue
        evaluated_changes += 1
        was_correct = mapping.get(before) == truth
        is_correct = mapping.get(after) == truth
        corrected += not was_correct and is_correct
        regressed += was_correct and not is_correct
    return dict(changed_words=changes, changed_evaluated_words=evaluated_changes,
                corrected_vs_mps=corrected, regressed_vs_mps=regressed)


def load_diagnostics(folder):
    data = {}
    for path in sorted(folder.glob("*/speaker-diagnostics.json")):
        d = json.loads(path.read_text())
        mode = d["speaker_identification"]["requested"]
        if mode not in MODES or mode in data:
            raise ValueError(f"Unexpected or duplicate diagnostic in {folder.name}")
        data[mode] = d
    if set(data) != set(MODES):
        raise ValueError(f"Expected three completed diagnostic files in {folder.name}")
    return data


def replay_clip(reference, diagnostics, expected):
    baseline_scores = compare(reference, diagnostics)
    for mode in MODES:
        if any(baseline_scores[mode][key] != expected[mode][key] for key in SCORE_FIELDS):
            raise ValueError(f"Stored baseline does not reproduce: {mode}")
    mps = diagnostics["mps"]
    labels = word_reference(mps["aligned_words"], reference, mps["seconds"]["audio"])
    start = time.perf_counter()
    proposals = candidates(diagnostics)
    replay_seconds = time.perf_counter() - start
    scores = {mode: {key: baseline_scores[mode][key] for key in (*SCORE_FIELDS, "accuracy")}
              for mode in MODES}
    for name, assigned in proposals.items():
        scores[name] = dict(word_score(labels, assigned),
                            **audit(labels, mps["assigned_speakers"], assigned))
    return dict(scores=scores, replay_seconds=replay_seconds)


def replay_run(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    baseline = json.loads((folder / "report.json").read_text())
    if manifest["split"] != "dev" or baseline["split"] != "dev":
        raise ValueError("Development replay only; keep the held-out test set untouched")
    expected = {row["clip"]: row["scores"] for row in baseline["clips"]}
    records = []
    seen = set()
    for item in manifest["clips"]:
        clip = Path(item["audio"]).stem
        reference_path = Path(item["rttm"])
        if clip in seen or clip not in expected or reference_path.stem != clip:
            raise ValueError("Manifest and baseline report disagree")
        seen.add(clip)
        result = replay_clip(read_rttm(reference_path),
                             load_diagnostics(folder / clip), expected[clip])
        records.append(dict(clip=clip, **result))
    if not records or seen != set(expected):
        raise ValueError("Benchmark is incomplete")
    aggregate = {}
    for name in MODES + CANDIDATES:
        rows = [r["scores"][name] for r in records]
        keys = SCORE_FIELDS + (("changed_words", "changed_evaluated_words",
                               "corrected_vs_mps", "regressed_vs_mps")
                              if name in CANDIDATES else ())
        scores = {key: sum(r[key] for r in rows) for key in keys}
        scores["accuracy"] = (scores["correct"] / scores["evaluated"]
                              if scores["evaluated"] else None)
        scores["net_correct_vs_mps"] = scores["correct"] - sum(
            r["scores"]["mps"]["correct"] for r in records)
        aggregate[name] = scores
    return dict(split="dev", clips=records, aggregate=aggregate,
                replay_seconds=sum(r["replay_seconds"] for r in records),
                schema_version=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed benchmark result directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    args = parser.parse_args()
    report = replay_run(args.run.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["aggregate"], indent=2, allow_nan=False))
    print(f"Replay report: {args.output}")


if __name__ == "__main__":
    main()
