"""Experimental two-source affine alignment; no ASR or app integration."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal

RATE = 1000
WINDOW = 2 * RATE
MAX_SECONDS = 7200


def decode(path):
    # Bound the prototype's memory use; never truncate a long source silently.
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(Path(path).resolve()),
        "-map", "0:a:0", "-vn", "-t", str(MAX_SECONDS + 1),
        "-ac", "1", "-ar", str(RATE), "-f", "f32le", "pipe:1",
    ])
    return np.frombuffer(raw, dtype="<f4")


def prepare(x):
    x = np.asarray(x, dtype=np.float32)
    if (x.ndim != 1 or not 10 * RATE <= len(x) <= MAX_SECONDS * RATE
            or not np.isfinite(x).all()):
        raise ValueError("Expected finite mono audio, 10 seconds to two hours")
    sos = signal.butter(3, [30, 250], btype="bandpass", fs=RATE, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def align(reference, source):
    """Return t_reference = offset_seconds + scale * t_source, or rejection."""
    a, b = prepare(reference), prepare(source)
    # Local normalisation prevents loud regions from dominating the search.
    sums = np.r_[0., np.cumsum(a, dtype=np.float64)]
    squares = np.r_[0., np.cumsum(a.astype(np.float64)**2)]
    energy = np.maximum(0, squares[WINDOW:] - squares[:-WINDOW]
                        - (sums[WINDOW:] - sums[:-WINDOW])**2 / WINDOW)
    norm = np.sqrt(energy).astype(np.float32)
    del sums, squares, energy
    hits = []
    for start in np.unique(np.linspace(0, len(b)-WINDOW, 17, dtype=int)):
        q = b[start:start+WINDOW].copy()
        q -= q.mean()
        qnorm = np.linalg.norm(q)
        if qnorm < 1e-5:
            continue
        scores = np.abs(signal.fftconvolve(a, q[::-1], mode="valid"))
        np.divide(scores, norm*qnorm, out=scores, where=norm > 1e-5)
        scores[norm <= 1e-5] = 0
        peak = int(np.argmax(scores))
        score = float(scores[peak])
        # Ignore the immediate neighbourhood, but reject distant competing peaks.
        radius = RATE // 10
        scores[max(0, peak-radius):peak+radius+1] = 0
        if score >= 0.35 and score - float(scores.max()) >= 0.10:
            hits.append(((start+WINDOW/2)/RATE, (peak+WINDOW/2)/RATE, score))
    def reject(reason):
        return {"accepted": False, "reason": reason, "anchors": len(hits)}
    if len(hits) < 4:
        return reject("insufficient distinctive matches")
    x, y, _ = np.asarray(hits).T
    scale, offset = np.polyfit(x, y, 1)
    residual = np.abs(y - (offset + scale*x))
    if abs(scale-1) > 0.002 or residual.max() > 0.05:
        return reject("non-affine or implausible clock mapping")
    lo = max(0., -offset/scale)
    hi = min(len(b)/RATE, (len(a)/RATE-offset)/scale)
    if hi-lo < 8 or np.ptp(x) < 0.5*(hi-lo):
        return reject("insufficient overlap coverage")
    # Require evidence near both ends and throughout the overlapping region.
    if max(x.min()-lo, hi-x.max(), np.diff(x).max()) > max(3., 0.25*(hi-lo)):
        return reject("unsupported interval within overlap")
    return {"accepted": True, "offset_seconds": float(offset),
            "scale": float(scale), "drift_ppm": float((scale-1)*1e6),
            "anchors": len(hits), "overlap_source_seconds": [float(lo), float(hi)],
            "max_anchor_residual_seconds": float(residual.max())}


def self_test():
    """Generated signals only, known ground-truth offsets and clock rates."""
    rng = np.random.default_rng(20261126)
    base = signal.sosfilt(signal.butter(3, [40, 200], btype="bandpass", fs=RATE,
                                       output="sos"), rng.normal(size=100*RATE))
    t = np.arange(len(base))/RATE
    def source(offset, scale, duration):
        return np.interp(offset+scale*np.arange(int(duration*RATE))/RATE, t, base)
    tests = [
        ("identical", base, base, 0, 1),
        ("positive offset", base, source(30, 1, 60), 30, 1),
        ("negative offset", base[20*RATE:], base[:75*RATE], -20, 1),
        ("positive drift", base, source(5, 1.0005, 85), 5, 1.0005),
        ("negative drift", base, source(5, .9995, 85), 5, .9995),
        ("gain, noise, inverted polarity", base,
         -0.4*source(10, 1.0002, 70)+rng.normal(0, .08, 70*RATE), 10, 1.0002),
        ("unrelated", base, rng.normal(size=60*RATE), None, None),
        ("silence", base, np.zeros(60*RATE), None, None),
        ("non-overlapping", base[:40*RATE], base[60*RATE:], None, None),
        ("repeated waveform", np.tile(base[:4*RATE], 25), base[:4*RATE].tolist()*15, None, None),
        ("internal edit", base, np.r_[base[:40*RATE], base[42*RATE:]], None, None),
    ]
    for name, a, b, offset, scale in tests:
        r = align(a, b)
        if offset is None:
            assert not r["accepted"], (name, r)
        else:
            assert r["accepted"], (name, r)
            lo, hi = r["overlap_source_seconds"]
            errors = [abs(r["offset_seconds"]+r["scale"]*u-offset-scale*u) for u in (lo, hi)]
            assert max(errors) <= .05, (name, errors, r)
        print(name, json.dumps(r, sort_keys=True))
    print(f"{len(tests)}/{len(tests)} synthetic tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", nargs="?")
    parser.add_argument("source", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.reference and args.source:
        try:
            result = align(decode(args.reference), decode(args.source))
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            parser.exit(2, f"Alignment input failed: {type(exc).__name__}\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result["accepted"] else 1)
    else:
        parser.error("provide two files or --self-test")
