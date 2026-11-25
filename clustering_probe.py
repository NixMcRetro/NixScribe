"""Developer-only offline Community-1 clustering experiment; no ASR or alignment."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

PINS = {"pyannote.audio": "4.0.7", "whisperx": "3.8.6", "torch": "2.8.0"}
MODEL = "pyannote/speaker-diarization-community-1"


class Memo:
    """One recording per adapter; never expose the stored value to mutation."""
    def __init__(self, fn):
        self.fn = fn
        self.value = None
        self.calls = self.hits = 0
        self.seconds = 0.0

    def __call__(self, *args, **kwargs):
        if self.value is None:
            start = time.perf_counter()
            value = self.fn(*args, **kwargs)
            self.seconds = time.perf_counter() - start
            self.calls += 1
            self.value = copy.deepcopy(value)
            return value
        self.hits += 1
        return copy.deepcopy(self.value)


def self_test():
    memo = Memo(lambda: {"data": [1, 2]})
    memo()["data"][0] = 9
    memo()["data"][1] = 8
    if memo() != {"data": [1, 2]} or (memo.calls, memo.hits) != (1, 2):
        raise RuntimeError("Cache isolation failed")
    print("Cache isolation self-test passed.")


def rows(annotation):
    return sorted((float(s.start), float(s.end), str(label))
                  for s, _, label in annotation.itertracks(yield_label=True))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      PYANNOTE_METRICS_ENABLED="0", PYTORCH_ENABLE_MPS_FALLBACK="0")
    os.umask(0o077)
    found = {key: importlib.metadata.version(key) for key in PINS}
    if found != PINS:
        raise RuntimeError(f"Unexpected dependency versions: {found}")
    import numpy as np
    import torch
    from huggingface_hub import try_to_load_from_cache
    from diarization_backend import require_mps
    require_mps()
    config = try_to_load_from_cache(MODEL, "config.yaml")
    if not isinstance(config, str) or not Path(config).is_file():
        raise RuntimeError("Community-1 config not cached; no download attempted")
    config = Path(config)
    print(f"Versions: {found}; cached snapshot: {config.parent.name}", flush=True)
    if args.preflight:
        print("Preflight passed; full model load not verified.")
        return
    if args.audio is None or args.output is None:
        parser.error("--audio and --output are required")
    if not args.audio.is_file() or args.output.exists():
        raise ValueError("Audio must exist and output directory must be new")
    from whisperx.audio import load_audio, SAMPLE_RATE
    from whisperx.diarize import DiarizationPipeline
    audio = load_audio(str(args.audio))
    waveform = torch.from_numpy(audio[None, :])
    start = time.perf_counter()
    # Hub offline mode permits only existing cached weights.
    p = DiarizationPipeline(model_name=MODEL, device="mps", token=None).model
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - start
    if type(p.clustering).__name__ != "VBxClustering":
        raise RuntimeError("Unexpected clustering implementation")
    cfg = {key: float(getattr(p.clustering, key)) for key in ("threshold", "Fa", "Fb")}
    cfg.update(constrained_assignment=bool(p.clustering.constrained_assignment),
               embedding_exclude_overlap=bool(p.embedding_exclude_overlap),
               segmentation_min_duration_off=float(p.segmentation.min_duration_off))
    spec = p._segmentation.model.specifications
    seg = Memo(p.get_segmentations)
    emb = Memo(p.get_embeddings)
    trials, outputs = [], {}

    def run(name, **kwargs):
        print(f"Running {name}...", flush=True)
        before = time.perf_counter()
        with torch.inference_mode():
            out = p({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
        torch.mps.synchronize()
        regular = rows(out.speaker_diarization)
        exclusive = rows(out.exclusive_speaker_diarization)
        result = dict(name=name, settings=kwargs, seconds=time.perf_counter() - before,
                      speakers=len(out.speaker_diarization.labels()),
                      intervals=len(regular), segmentation_calls=seg.calls,
                      embedding_calls=emb.calls)
        outputs[name] = (regular, exclusive, out.speaker_embeddings)
        trials.append(result)
        print(json.dumps(result), flush=True)
        return result

    # Reuse neural results only, keeping upstream clustering and reconstruction intact.
    with patch.object(p, "get_segmentations", seg), patch.object(p, "get_embeddings", emb):
        baseline = run("auto_capture")
        if not baseline["speakers"] or (seg.calls, emb.calls) != (1, 1):
            raise RuntimeError("Incomplete capture or no speech")
        run("auto_replay")
        a, b = outputs["auto_capture"], outputs["auto_replay"]
        equivalent = (a[:2] == b[:2] and np.array_equal(a[2], b[2], equal_nan=True))
        if not equivalent or (seg.calls, emb.calls, seg.hits, emb.hits) != (1, 1, 1, 1):
            raise RuntimeError("Cached baseline differs; no candidates attempted")
        n = baseline["speakers"]
        for k in sorted({n - 1, n, n + 1} & set(range(1, 11))):
            run(f"count_{k}", num_speakers=k)
        if (seg.calls, emb.calls, seg.hits, emb.hits) != (1, 1, len(trials)-1, len(trials)-1):
            raise RuntimeError("Unexpected neural-stage calls")
    s, e = seg.value, emb.value
    sw = s.sliding_window
    if e.shape[:2] != (s.data.shape[0], s.data.shape[2]):
        raise RuntimeError("Embedding and segmentation shape mismatch")
    args.output.mkdir(mode=0o700, exist_ok=False)
    cache = args.output / "private-inference.npz"
    with cache.open("xb") as f:
        np.savez_compressed(f, segmentation=s.data, embeddings=e,
                           window=np.array([sw.start, sw.duration, sw.step]))
    with (args.output / "private-intervals.json").open("x") as f:
        json.dump({k: {"regular": v[0], "exclusive": v[1]} for k, v in outputs.items()},
                  f, allow_nan=False)
    report = dict(schema_version=1, clip=args.audio.stem, versions=found,
                  model=MODEL, snapshot=config.parent.name, config_sha256=sha(config),
                  audio_sha256=sha(args.audio), device="mps", config=cfg,
                  segmentation_model=type(p._segmentation.model).__name__,
                  powerset=bool(spec.powerset), segmentation_shape=list(s.data.shape),
                  embedding_shape=list(e.shape), window_seconds=[sw.start, sw.duration, sw.step],
                  model_load_seconds=load_seconds, segmentation_seconds=seg.seconds,
                  embedding_seconds=emb.seconds, cache_bytes=cache.stat().st_size,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  exact_cached_baseline=bool(equivalent), trials=trials)
    with (args.output / "probe-report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print("Share only probe-report.json; keep private arrays and intervals locally.", flush=True)
