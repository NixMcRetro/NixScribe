"""Developer-only, local comparison of NixScribe speaker identification modes."""
import argparse
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import time
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
import wave

MODES = ("coreml", "mps", "combined")


def read_rttm(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER" or fields[1] != path.stem:
            raise ValueError(f"Unexpected RTTM row in {path.name}")
        start, duration = float(fields[3]), float(fields[4])
        rows.append(dict(start=start, end=start + duration, speaker=fields[7]))
    if not rows:
        raise ValueError(f"No reference speech: {path.name}")
    return rows


def checked(rows, duration):
    out = []
    for row in rows:
        a, b, speaker = row["start"], row["end"], row["speaker"]
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b
                and isinstance(speaker, str) and speaker):
            raise ValueError("Invalid interval")
        if a < duration and speaker != "UNKNOWN":
            out.append((a, min(b, duration), speaker))
    return out


def best_mapping(weights):
    """Exact one-to-one maximum-weight assignment; labels are anonymous."""
    left = sorted({a for a, b in weights})
    right = sorted({b for a, b in weights})
    # shortcut: exact subset search is small for short 2-4 speaker clips;
    # use a polynomial assignment algorithm before evaluating large meetings.
    if len(right) > 12 or len(left) > 12:
        raise ValueError("Benchmark supports up to 12 labels per recording")

    @lru_cache(None)
    def solve(i, mask):
        if i == len(left):
            return 0, ()
        score, tail = solve(i + 1, mask)
        best = score, (None,) + tail
        for j, b in enumerate(right):
            w = weights.get((left[i], b), 0)
            if not w or mask & (1 << j):
                continue
            score, tail = solve(i + 1, mask | (1 << j))
            if w + score > best[0]:
                best = w + score, (b,) + tail
        return best

    return dict(zip(left, solve(0, 0)[1]))


def interval_score(reference, hypothesis, duration):
    """Whole-clip, overlap-inclusive DER, no forgiveness collar."""
    ref, hyp = checked(reference, duration), checked(hypothesis, duration)
    events = sorted({0, duration} | {t for a, b, _ in ref + hyp for t in (a, b)})
    frames, weights = [], Counter()
    for a, b in zip(events, events[1:]):
        r = {s for x, y, s in ref if x < b and y > a}
        h = {s for x, y, s in hyp if x < b and y > a}
        dt = b - a
        frames.append((dt, r, h))
        for x in h:
            for y in r:
                weights[x, y] += dt
    mapping = best_mapping(weights)
    total = miss = false_alarm = confusion = 0.0
    for dt, r, h in frames:
        total += dt * len(r)
        miss += dt * max(0, len(r) - len(h))
        false_alarm += dt * max(0, len(h) - len(r))
        confusion += dt * (min(len(r), len(h)) - sum(mapping.get(x) in r for x in h))
    return dict(reference_speaker_seconds=total, missed_seconds=miss,
                false_alarm_seconds=false_alarm, confusion_seconds=confusion,
                der=(miss + false_alarm + confusion) / total if total else None)


def word_reference(words, reference, duration):
    """Score only words with >=50% coverage by exactly one reference speaker."""
    rows = checked(reference, duration)
    labels = []
    for w in words:
        a, b = w.get("start"), w.get("end")
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration):
            labels.append(None)
            continue
        overlapping = [(max(a, x), min(b, y), s) for x, y, s in rows if x < b and y > a]
        speakers = {s for _, _, s in overlapping}
        coverage, last = 0.0, a
        for x, y, _ in sorted(overlapping):
            coverage += max(0, y - max(last, x))
            last = max(last, y)
        labels.append(next(iter(speakers)) if len(speakers) == 1
                      and coverage >= (b - a) / 2 else None)
    return labels


def word_score(labels, assigned):
    if len(labels) != len(assigned):
        raise ValueError("Speaker count does not match aligned words")
    pairs = [(h, r) for h, r in zip(assigned, labels) if r is not None]
    mapping = best_mapping(Counter((h, r) for h, r in pairs if h != "UNKNOWN"))
    correct = sum(mapping.get(h) == r for h, r in pairs)
    return dict(correct=correct, evaluated=len(pairs), excluded=len(labels) - len(pairs),
                unknown=sum(h == "UNKNOWN" for h, _ in pairs),
                accuracy=correct / len(pairs) if pairs else None)


def compare(reference, diagnostics):
    base = diagnostics["combined"]
    words = base["aligned_words"]
    duration = base["seconds"]["audio"]
    labels = word_reference(words, reference, duration)
    results = {}
    for mode in MODES:
        d = diagnostics[mode]
        identity = d["speaker_identification"]
        if (d["schema_version"] != 1 or identity["requested"] != mode
                or identity["actual"] != mode or identity["degraded"]
                or d["recognition"]["actual"] != "mlx-large-v3"):
            raise ValueError(f"Fallback or incompatible diagnostics: {mode}")
        if (d["aligned_words"] != words or d["seconds"]["audio"] != duration
                or d["versions"] != base["versions"] or d["models"] != base["models"]):
            raise ValueError(f"Incompatible word alignment or model versions: {mode}")
        row = dict(word_score(labels, d["assigned_speakers"]), seconds=d["seconds"])
        if mode != "combined":
            row["raw_interval_score"] = interval_score(reference, d["raw_intervals"][mode], duration)
        results[mode] = row
    results["combined"]["uncertain_words"] = sum(
        x["uncertain"] for x in base["reconciliation"]["decisions"])
    return results


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def api(port, endpoint, payload=None):
    request = Request(f"http://127.0.0.1:{port}/api/{endpoint}",
                      data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=30) as response:
        return json.load(response)


def run_job(port, audio, projects, mode):
    if api(port, "status")["running"]:
        raise RuntimeError("NixScribe is busy; do not use the UI during benchmarking")
    before = time.monotonic()
    response = api(port, "start", dict(input_file=str(audio), projects_dir=str(projects),
                   model="mlx-large-v3", speaker_count="auto", diarization_device=mode,
                   save_diagnostics=True, copy_original=False, save_mp3=False))
    if not response["ok"]:
        raise RuntimeError("NixScribe did not start the job")
    while True:
        status = api(port, "status")
        if not status["running"]:
            break
        time.sleep(1)
    if status["status"] != "Complete":
        raise RuntimeError(f"Job did not complete: {mode}")
    project = Path(status["project_dir"]).resolve()
    if not project.is_relative_to(projects) or Path(status["input_file"]).resolve() != audio:
        raise RuntimeError("Unexpected job output; stop benchmark")
    data = json.loads((project / "speaker-diagnostics.json").read_text())
    data["wall_seconds"] = time.monotonic() - before
    return data


def select_clips(audio_dir, rttm_dir, per_group):
    wavs = {}
    for path in audio_dir.rglob("*.wav"):
        if path.stem in wavs:
            raise ValueError(f"Duplicate audio ID: {path.stem}")
        wavs[path.stem] = path.resolve()
    groups = {2: [], 3: [], 4: []}
    for path in sorted(rttm_dir.glob("*.rttm")):
        if path.stem not in wavs:
            continue
        ref = read_rttm(path)
        count = len({r["speaker"] for r in ref})
        if count not in groups or len(groups[count]) >= per_group:
            continue
        audio = wavs[path.stem]
        with wave.open(str(audio), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
        if 30 <= duration <= 180:
            groups[count].append((audio, path.resolve()))
    if any(len(group) != per_group for group in groups.values()):
        raise ValueError("Not enough matching 30-180s clips with 2, 3 and 4 speakers")
    return [pair for group in groups.values() for pair in group]


def aggregate(records):
    result = {}
    for mode in MODES:
        rows = [record["scores"][mode] for record in records]
        n = sum(r["evaluated"] for r in rows)
        correct = sum(r["correct"] for r in rows)
        entry = dict(correct=correct, evaluated=n, accuracy=correct / n if n else None,
                     excluded=sum(r["excluded"] for r in rows),
                     unknown=sum(r["unknown"] for r in rows),
                     wall_seconds=sum(r["wall_seconds"] for r in rows))
        if mode != "combined":
            metrics = [r["raw_interval_score"] for r in rows]
            fields = ("reference_speaker_seconds", "missed_seconds", "false_alarm_seconds",
                      "confusion_seconds")
            raw = {k: sum(r[k] for r in metrics) for k in fields}
            raw["der"] = sum(raw[k] for k in fields[1:]) / raw[fields[0]] if raw[fields[0]] else None
            entry["raw_interval_score"] = raw
        result[mode] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--rttm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--per-group", type=int, default=1)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.per_group <= 10:
        parser.error("--per-group must be 1-10")
    clips = select_clips(args.audio_dir, args.rttm_dir, args.per_group)
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = [{"audio": str(a), "rttm": str(r)} for a, r in clips]
    (output / "manifest.json").write_text(json.dumps({"split": args.split, "clips": manifest}, indent=2))
    records = []
    for index, (audio, rttm) in enumerate(clips):
        projects = output / audio.stem
        projects.mkdir()
        diagnostics = {}
        for mode in MODES[index % 3:] + MODES[:index % 3]:
            print(f"{audio.stem}: {mode}", flush=True)
            diagnostics[mode] = run_job(args.port, audio, projects, mode)
        scores = compare(read_rttm(rttm), diagnostics)
        for mode in MODES:
            scores[mode]["wall_seconds"] = diagnostics[mode]["wall_seconds"]
        records.append(dict(clip=audio.stem, scores=scores))
        report = dict(split=args.split, clips=records, aggregate=aggregate(records))
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
