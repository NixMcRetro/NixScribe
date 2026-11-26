"""Offline tests of frozen selection and production cache adapter; no weights."""
import copy
import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import intelligent_auto as auto


class Annotation:
    def __init__(self, spans):
        self.spans = spans

    def itertracks(self, yield_label=True):
        for a, b, label in self.spans:
            yield types.SimpleNamespace(start=a, end=b), None, label

    def labels(self):
        return sorted({s for _, _, s in self.spans})


class VBxClustering:
    threshold = .6
    Fa = .07
    Fb = .8
    constrained_assignment = True

    def filter_embeddings(self, embeddings, segmentations=None, min_active_ratio=.20):
        return [1, 2], None


class FakePipeline:
    def __init__(self):
        self.clustering = VBxClustering()
        self.embedding_exclude_overlap = True
        self.segmentation = types.SimpleNamespace(min_duration_off=0.)
        self._segmentation = types.SimpleNamespace(model=types.SimpleNamespace(
            specifications=types.SimpleNamespace(powerset=True)))
        self.candidates = []

    def get_segmentations(self):
        raise AssertionError('Uncached segmentation')

    def get_embeddings(self):
        raise AssertionError('Uncached embeddings')

    def __call__(self, inp):
        s, e = self.get_segmentations(), self.get_embeddings()
        self.clustering.filter_embeddings(e, segmentations=s)
        self.candidates.append((self.clustering.threshold, self.clustering.Fa,
                                self.clustering.Fb))
        return types.SimpleNamespace(speaker_diarization=Annotation([(0., 1., 'A'), (1., 2., 'B')]))


class IntelligentAutoTests(unittest.TestCase):
    def test_consensus_is_label_invariant_and_prefers_baseline_ties(self):
        words = [{'start': 0., 'end': .5}, {'start': .5, 'end': 1.}]
        assignments = {name: [f'{i}A', f'{i}B'] for i, name in enumerate(auto.candidate_names)}
        winner, distances, _ = auto.consensus_select(assignments, words)
        self.assertEqual(winner, 'baseline')
        self.assertTrue(all(x == 0. for x in distances.values()))

    def test_five_candidates_reuse_cache_and_restore_defaults(self):
        p = FakePipeline()
        seg, emb = auto.Cached([1]), auto.Cached([2])
        words = [{'start': .1, 'end': .8}, {'start': 1.1, 'end': 1.8}]
        baseline = [[0., 1., 'A'], [1., 2., 'B']]
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(inference_mode=nullcontext)}):
            result = auto.predict_cached(p, {}, seg, emb, words, [0]*32000, [], 2., baseline)
        self.assertEqual(result['selected'], 'baseline')
        self.assertEqual(result['labels'], ['A', 'B'])
        self.assertEqual((seg.hits, emb.hits), (5, 5))
        self.assertEqual(p.candidates, [v[1:4] for v in auto.VARIANTS])
        self.assertEqual({k: getattr(p.clustering,k) for k in auto.DEFAULTS}, auto.DEFAULTS)

    def test_baseline_mismatch_rejected_and_defaults_restored(self):
        p = FakePipeline()
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(inference_mode=nullcontext)}):
            with self.assertRaisesRegex(RuntimeError, 'baseline differs'):
                auto.predict_cached(p, {}, auto.Cached([]), auto.Cached([]), [], [], [], 2., [])
        self.assertEqual({k: getattr(p.clustering,k) for k in auto.DEFAULTS}, auto.DEFAULTS)

    def test_label_application_preserves_transcription_fields(self):
        result = {'segments': [{'text': 'Hello world', 'start': 0., 'end': 2., 'words': [
            {'word': 'Hello', 'start': 0., 'end': .5}, {'word': 'world', 'start': 1., 'end': 2.}]}]}
        before = copy.deepcopy(result)
        with self.assertRaises(ValueError):
            auto.apply_labels(result, ['A'])
        self.assertEqual(result, before)
        auto.apply_labels(result, ['A', 'B'])
        self.assertEqual([w.pop('speaker') for w in result['segments'][0]['words']], ['A', 'B'])
        self.assertEqual(result['segments'][0].pop('speaker'), 'UNKNOWN')
        self.assertEqual(result, before)

    def test_cache_value_isolation(self):
        cached = auto.Cached({'items': [1]})
        cached()['items'][0] = 2
        self.assertEqual(cached(), {'items': [1]})


if __name__ == '__main__':
    unittest.main()
