"""Frozen reference-free Intelligent Auto v5 algorithm and isolated worker adapter.

Research source archive SHA-256:
 b7ab64c45ed4c0e26344490bec190ea9c2ea656b3b073df5051425202a5bb029
The assignment, consensus, mapping and acoustic functions are extracted verbatim
from the archived v5, benchmark and acoustic sources (see validation guide).
No references, audio exports, transcripts or caches are written by this module.
"""
from array import array
from collections import Counter, defaultdict
from functools import lru_cache
import copy
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import statistics as st
import sys
import wave
from unittest.mock import patch

VARIANTS = (
    ("baseline", .60, .07, .80, .20),
    ("original_clean_15", .60, .07, .80, .15),
    ("conservative_clean_10", .60, .07, .70, .10),
    ("conservative_clean_20", .60, .07, .70, .20),
    ("accuracy_clean_20", .55, .08, .60, .20),
)
candidate_names = tuple(v[0] for v in VARIANTS)
MODEL = "pyannote/speaker-diarization-community-1"
SNAPSHOT = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
CONFIG_SHA = "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be"
VERSIONS = {"pyannote.audio": "4.0.7", "torch": "2.8.0", "numpy": "2.5.5",
            "scipy": "1.18.1", "scikit-learn": "1.9.1", "whisperx": "3.8.6"}
DEFAULTS = {"threshold": .6, "Fa": .07, "Fb": .8, "constrained_assignment": True}

CANDIDATES = ('acoustic_short', 'boundary_verified', 'adaptive_verified', 'combined_verified')
N = 512
RATE = 16000
REVERSE = [int(f'{i:09b}'[::-1], 2) for i in range(N)]
WINDOW = [0.54 - 0.46 * math.cos(2 * math.pi * i / 399) for i in range(400)]
TWIDDLES = {n: [complex(math.cos(-2*math.pi*i/n), math.sin(-2*math.pi*i/n))
                for i in range(n//2)] for n in (2,4,8,16,32,64,128,256,512)}
MEL = [700*(10**((2595*math.log10(1+80/700)+i/21*(2595*math.log10(1+7600/700)-2595*math.log10(1+80/700)))/2595)-1) for i in range(22)]
FILTERS = [[(k, max(0, min((k*RATE/N-a)/(b-a), (c-k*RATE/N)/(c-b))))
            for k in range(1,N//2+1) if a < k*RATE/N < c]
           for a,b,c in zip(MEL,MEL[1:],MEL[2:])]
DCT = [[math.cos(math.pi*j*(k+.5)/20) for k in range(20)] for j in range(1,13)]


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


def rows(annotation):
    return sorted([
        [float(s.start), float(s.end), str(label)]
        for s, _, label in annotation.itertracks(
            yield_label=True
        )
    ])


def assign(words, annotation):
    intervals = sorted(
        [(s.start, s.end, str(label))
         for s, _, label in annotation.itertracks(
             yield_label=True
         )],
        key=lambda x: x[0]
    )
    result = []

    for word in words:
        a = word.get("start")
        if a is None or not intervals:
            result.append("UNKNOWN")
            continue

        b = word.get("end", a)
        overlap = {}

        for x, y, speaker in intervals:
            if x >= b:
                break
            amount = min(y, b) - max(x, a)
            if amount > 0:
                overlap[speaker] = (
                    overlap.get(speaker, 0) + amount
                )

        if overlap:
            result.append(max(overlap, key=overlap.get))
        else:
            midpoint = (a + b) / 2
            result.append(min(
                intervals,
                key=lambda r: abs(
                    (r[0] + r[1]) / 2 - midpoint
                )
            )[2])

    return result


class Cached:
    def __init__(self, data):
        self.data = data
        self.hits = 0

    def __call__(self, *args, **kwargs):
        self.hits += 1
        return copy.deepcopy(self.data)


def disagreement(a, b):
    # Compare anonymous labels without reference annotations.
    weights = Counter(
        (x, y) for x, y in zip(a, b)
        if x != "UNKNOWN" and y != "UNKNOWN"
    )
    mapping = best_mapping(weights)
    matches = sum(
        x != "UNKNOWN" and y != "UNKNOWN"
        and mapping.get(x) == y
        for x, y in zip(a, b)
    )
    return 1 - matches / len(a)


def consensus_select(assignments, words):
    # Use all valid timed words, not reference-scored words.
    indices = [
        i for i, w in enumerate(words)
        if isinstance(w.get("start"), (int, float))
        and isinstance(w.get("end"), (int, float))
        and math.isfinite(w["start"])
        and math.isfinite(w["end"])
        and w["start"] < w["end"]
    ]
    assert indices

    vectors = {
        name: [values[i] for i in indices]
        for name, values in assignments.items()
    }

    distances = {
        name: {
            other: disagreement(
                vectors[name], vectors[other]
            )
            for other in candidate_names
            if name != other
        }
        for name in candidate_names
    }

    mean_distance = {
        name: sum(values.values()) / len(values)
        for name, values in distances.items()
    }

    # Consensus medoid; prefer original baseline on ties.
    winner = min(
        candidate_names,
        key=lambda name: (
            round(mean_distance[name], 12),
            candidate_names.index(name)
        )
    )

    return winner, mean_distance, distances


def read_samples(path, duration):
    with wave.open(str(path), "rb") as w:
        assert (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate()
        ) == (1, 2, 16000)

        assert abs(
            w.getnframes() / 16000 - duration
        ) <= 0.05

        samples = array("h")
        samples.frombytes(w.readframes(w.getnframes()))

        if sys.byteorder != "little":
            samples.byteswap()

    return samples


def frame(samples, center):
    start = round(center*RATE)-200
    if start < 1 or start+400 > len(samples):
        return None
    s = samples[start:start+400]
    if sum(x*x for x in s)/400 < 60**2:
        return None
    x = [(v-.97*samples[start+i-1])*WINDOW[i] for i,v in enumerate(s)] + [0]*112
    a = [complex(x[i]) for i in REVERSE]
    for n, tw in TWIDDLES.items():
        half = n//2
        for base in range(0,N,n):
            for j,t in enumerate(tw):
                u,v = a[base+j],a[base+j+half]*t
                a[base+j],a[base+j+half]=u+v,u-v
    power = [v.real*v.real+v.imag*v.imag for v in a[:N//2+1]]
    logs = [math.log(max(1.,sum(power[k]*weight for k,weight in filt))) for filt in FILTERS]
    return [sum(x*y for x,y in zip(logs,coefs)) for coefs in DCT]


def feature(samples, a, b):
    if b-a < .4:
        return None
    # evenly spaced interior frames; no frames cross the segment boundary
    count = min(16,max(4,round((b-a)*10)))
    frames = [frame(samples,a+.013+(b-a-.026)*(i+.5)/count) for i in range(count)]
    frames = [x for x in frames if x is not None]
    if len(frames) < 4:
        return None
    return [st.mean(x) for x in zip(*frames)]


def intervals(rows):
    out=[]
    for r in rows:
        a,b,s=r['start'],r['end'],r['speaker']
        if not (math.isfinite(a) and math.isfinite(b) and 0<=a<b and isinstance(s,str) and s):
            raise ValueError('Invalid interval')
        if s!='UNKNOWN': out.append((a,b,s))
    return sorted(out)


def timing(w,duration):
    a,b=w.get('start'),w.get('end')
    if isinstance(a,(int,float)) and isinstance(b,(int,float)) and 0<=a<b<=duration: return a,b
    return None


def exclusive(rows,a,b,min_fraction=.95):
    spans=[(max(a,x),min(b,y),s) for x,y,s in rows if x<b and y>a]
    labels={s for x,y,s in spans}
    if len(labels)!=1: return None
    covered=0; last=a
    for x,y,s in sorted(spans):
        covered+=max(0,y-max(last,x));last=max(last,y)
    return next(iter(labels)) if covered>=(b-a)*min_fraction else None


def map_engines(core,mps):
    weights=Counter()
    for a,b,s in core:
        for x,y,t in mps:
            overlap=min(b,y)-max(a,x)
            if overlap>0: weights[s,t]+=overlap
    mapping={}
    for s,t in weights:
        left=sorted((v,k[1]) for k,v in weights.items() if k[0]==s)
        right=sorted((v,k[0]) for k,v in weights.items() if k[1]==t)
        if left[-1][1]==t and right[-1][1]==s and (len(left)==1 or left[-1][0]>left[-2][0]) and (len(right)==1 or right[-1][0]>right[-2][0]): mapping[s]=t
    return mapping


def profiles(samples,core,mps,mapping):
    # Agreement only; exclude short MPS segments, overlapping speech, edges.
    train=defaultdict(list)
    for a,b,s in mps:
        if b-a<2.5: continue
        for i in range(math.floor(b-a-.4)):
            x=a+.2+i; y=x+1
            if y>b-.2 or exclusive(mps,x,y)!=s or mapping.get(exclusive(core,x,y))!=s: continue
            f=feature(samples,x,y)
            if f is not None: train[s].append((x,y,f))
    return train


def verify(samples,train,a,b,old,allowed):
    f=feature(samples,a,b)
    if f is None: return None
    # Prevent evaluation samples leaking into the profiles, even for disputed words.
    groups={s:[v for x,y,v in rows if y<=a-.1 or x>=b+.1] for s,rows in train.items()}
    groups={s:v for s,v in groups.items() if len(v)>=3}
    if old not in groups or len(groups)<2: return None
    all_features=[v for rows in groups.values() for v in rows]
    scale=[max(1.,st.pstdev(col)) for col in zip(*all_features)]
    distances={s: math.sqrt(sum(((x-st.median(col))/d)**2 for x,col,d in zip(f,zip(*rows),scale))) for s,rows in groups.items()}
    ranked=sorted(distances,key=distances.get); new=ranked[0]
    # A 20% margin against both the old speaker and every alternative.
    if new!=old and new in allowed and distances[new]<.8*distances[ranked[1]]:
        return new
    return None


def proposals(samples,words,baseline,core,mps,duration):
    mapping=map_engines(core,mps); train=profiles(samples,core,mps,mapping)
    result={name:list(baseline) for name in CANDIDATES}
    times=[timing(w,duration) for w in words]
    for i,(a,b,old) in enumerate(mps):
        if not .4<=b-a<=1.5 or exclusive(mps,a,b)!=old: continue
        # Do not erase potential cross-talk (multiple active Core ML labels).
        if len({s for x,y,s in core if x<b and y>a})>1: continue
        near=set()
        before=mps[i-1] if i else None; after=mps[i+1] if i+1<len(mps) else None
        if before and a-before[1]<=.25 and before[1]<=a and before[1]-before[0]>=2: near.add(before[2])
        if after and after[0]-b<=.25 and after[0]>=b and after[1]-after[0]>=2: near.add(after[2])
        # Interior boundary repair requires both neighbours to agree; edge repair one.
        boundary=near if len(near)==1 and ((before is None or after is None) or (before[2]==after[2] and a-before[1]<=.25 and after[0]-b<=.25)) else set()
        new=verify(samples,train,a,b,old,set(train)-{old})
        if new is None: continue
        for j,t in enumerate(times):
            if t and baseline[j]==old and t[0]>=a-.02 and t[1]<=b+.02:
                result['acoustic_short'][j]=new
                if new in boundary: result['boundary_verified'][j]=new
    # For disagreement, only consider short contiguous runs of words with
    # exclusive coverage by different mapped speakers in the two engines.
    runs=[]
    for j,t in enumerate(times):
        if t is None: continue
        a,b=t; old=baseline[j]; new=mapping.get(exclusive(core,a,b,.8))
        if not new or old==new or exclusive(mps,a,b,.8)!=old: continue
        if runs and runs[-1][2:4]==[old,new] and j==runs[-1][4][-1]+1 and a-runs[-1][1]<=.2 and b-runs[-1][0]<=2:
            runs[-1][1]=b; runs[-1][4].append(j)
        else: runs.append([a,b,old,new,[j]])
    for a,b,old,new,indices in runs:
        if exclusive(mps,a,b,.7)!=old or mapping.get(exclusive(core,a,b,.7))!=new: continue
        if verify(samples,train,a,b,old,{new})==new:
            for j in indices: result['adaptive_verified'][j]=new
    for j,old in enumerate(baseline):
        options={result[name][j] for name in ('boundary_verified','adaptive_verified')}-{old}
        if len(options)==1: result['combined_verified'][j]=next(iter(options))
    return result,{s:len(rows) for s,rows in train.items()}


def validate_pipeline(p):
    """Reject different versions/defaults instead of silently changing the experiment."""
    if type(p.clustering).__name__ != "VBxClustering":
        raise RuntimeError("Unsupported clustering implementation")
    if ({k: getattr(p.clustering, k) for k in DEFAULTS} != DEFAULTS
            or not p.embedding_exclude_overlap or p.segmentation.min_duration_off != 0
            or not p._segmentation.model.specifications.powerset):
        raise RuntimeError("Unexpected Community-1 defaults")


def validate_environment(p):
    from huggingface_hub import try_to_load_from_cache
    if {k: version(k) for k in VERSIONS} != VERSIONS:
        raise RuntimeError("Experimental dependency versions differ from validation")
    config = try_to_load_from_cache(MODEL, "config.yaml")
    if not isinstance(config, str):
        raise RuntimeError("Community-1 configuration is not cached")
    config = Path(config)
    with config.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if config.parent.name != SNAPSHOT or digest != CONFIG_SHA:
        raise RuntimeError("Community-1 snapshot differs from validation")
    validate_pipeline(p)


def predict_cached(p, inp, seg, emb, words, samples, core, duration, expected_baseline):
    """CPU reconstruction from a single in-memory capture, exactly five frozen candidates."""
    import torch
    validate_pipeline(p)
    original_filter = type(p.clustering).filter_embeddings
    assignments, candidates = {}, {}
    initial_hits = (seg.hits, emb.hits)
    try:
        with patch.object(p, "get_segmentations", seg), patch.object(p, "get_embeddings", emb):
            for name, threshold, fa, fb, ratio in VARIANTS:
                for key, value in DEFAULTS.items():
                    setattr(p.clustering, key, value)
                p.clustering.threshold = threshold
                p.clustering.Fa = fa
                p.clustering.Fb = fb
                counts = []

                def filtered(self, embeddings, segmentations=None, min_active_ratio=0.20):
                    result = original_filter(self, embeddings, segmentations=segmentations,
                                             min_active_ratio=ratio)
                    counts.append(len(result[0]))
                    if counts[-1] < 2:
                        raise RuntimeError("Insufficient training embeddings")
                    return result

                with patch.object(type(p.clustering), "filter_embeddings", new=filtered):
                    with torch.inference_mode():
                        result = p(dict(inp))
                if len(counts) != 1 or not 0 < len(result.speaker_diarization.labels()) <= 12:
                    raise RuntimeError("Unsupported clustering result")
                reg = rows(result.speaker_diarization)
                if name == "baseline" and reg != expected_baseline:
                    raise RuntimeError("Cached baseline differs from original")
                assignments[name] = assign(words, result.speaker_diarization)
                candidates[name] = reg
        if (seg.hits - initial_hits[0], emb.hits - initial_hits[1]) != (5, 5):
            raise RuntimeError("Unexpected cache replay call count")
    finally:
        for key, value in DEFAULTS.items():
            setattr(p.clustering, key, value)
    winner, stability, distances = consensus_select(assignments, words)
    adjusted, _ = proposals(samples, words, assignments[winner], intervals(core),
                            [tuple(row) for row in candidates[winner]], duration)
    labels = adjusted["acoustic_short"]
    if len(labels) != len(words) or not set(labels) <= set(assignments[winner]):
        raise RuntimeError("Acoustic stage changed label inventory")
    return {"selected": winner, "labels": labels, "intervals": candidates[winner],
            "candidate_assignments": assignments, "candidate_intervals": candidates,
            "mean_disagreement": stability, "pairwise_disagreement": distances,
            "acoustic_changes": sum(a != b for a, b in zip(labels, assignments[winner]))}


def apply_labels(result, labels):
    words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    if len(words) != len(labels):
        raise ValueError("Unexpected Intelligent Auto word count")
    for word, label in zip(words, labels):
        word["speaker"] = label
    # Do not attribute untimed/unmatched speech to a guessed segment speaker.
    for segment in result.get("segments", []):
        segment["speaker"] = "UNKNOWN"


def capture_and_predict(pipeline, audio, words, core, duration, report):
    """Capture on MPS, replay on CPU. If replay fails, return captured baseline."""
    import torch
    import numpy as np
    from clustering_probe import Memo
    p = pipeline.model
    # Before capture, a version/config mismatch uses the original worker path.
    try:
        validate_environment(p)
        samples = read_samples(audio, duration)
        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32)[None, :] / 32768.)
        if not len(samples):
            raise RuntimeError("Empty waveform")
    except Exception as exc:
        df = pipeline(str(audio), progress_callback=report)
        return {"intervals": df[["start", "end", "speaker"]].to_dict(orient="records"),
                "labels": None, "fallback_reason": type(exc).__name__}
    inp = {"waveform": waveform, "sample_rate": RATE}
    seg, emb = Memo(p.get_segmentations), Memo(p.get_embeddings)
    report(5.0)
    with patch.object(p, "get_segmentations", seg), patch.object(p, "get_embeddings", emb):
        with torch.inference_mode():
            captured = p(dict(inp))
        torch.mps.synchronize()
    baseline = rows(captured.speaker_diarization)
    def dictionaries(reg):
        return [dict(start=a, end=b, speaker=s) for a, b, s in reg]
    report(75.0)
    try:
        if (seg.calls, emb.calls) != (1, 1) or not baseline:
            raise RuntimeError("Incomplete capture")
        p.to(torch.device("cpu"))
        replay = predict_cached(p, inp, Cached(seg.value), Cached(emb.value), words,
                                samples, core, duration, baseline)
        report(100.0)
        return {"intervals": dictionaries(replay["intervals"]), "labels": replay["labels"],
                "selected": replay["selected"], "acoustic_changes": replay["acoustic_changes"],
                "segmentation_calls": seg.calls, "embedding_calls": emb.calls}
    except Exception as exc:
        # No second neural pass when candidate replay or acoustic adjustment fails.
        return {"intervals": dictionaries(baseline), "labels": None,
                "fallback_reason": type(exc).__name__}
