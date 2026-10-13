"""Pure tests for NixScribe transcripts and subtitles. No ML downloads needed."""
import re
import tempfile
import unittest
from pathlib import Path

from transcript_format import write_srt, write_txt


class TranscriptExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dest = Path(temp.name)
        self.media = Path("sample.m4a")
        self.data = {"segments": [
            {"speaker": "SPEAKER_00", "start": 0, "end": 3,
             "text": "Welcome to our conversation about the design process and how this works today."},
            {"speaker": "SPEAKER_00", "start": 3.2, "end": 5,
             "text": "We have a follow-up question."},
            {"speaker": "SPEAKER_01", "start": 5.2, "end": 10.5,
             "text": "Absolutely. The answer is long enough to need several subtitle cues over time, with accurate word boundaries."},
            {"speaker": None, "start": 10.6, "end": 11.2, "text": "Thanks."},
        ]}

    def test_txt_no_timestamps_and_speaker_turns(self):
        text = write_txt(self.data, self.media, self.dest).read_text()
        self.assertNotIn("-->", text)
        self.assertEqual(text.count("[SPEAKER_00]"), 1)
        self.assertIn("[SPEAKER_01]", text)
        self.assertIn("[UNKNOWN]", text)

    def test_long_speaker_turn_is_split_into_readable_paragraphs(self):
        # A single long WhisperX segment must not turn into a giant paragraph.
        sentence = "The explanation continues through another detailed step of this technical discussion."
        speech = " ".join([sentence] * 32)
        data = {"segments": [{
            "speaker": "SPEAKER_01", "start": 1.0, "end": 210.0,
            "text": speech,
        }]}
        txt = write_txt(data, self.media, self.dest).read_text()
        paragraphs = [p for p in txt.strip().split("\n\n") if p]
        self.assertGreaterEqual(len(paragraphs), 4)
        self.assertTrue(all(p.startswith("[SPEAKER_01] ") for p in paragraphs))
        self.assertTrue(all(len(p.split()) <= 91 for p in paragraphs))
        self.assertEqual(
            " ".join(p.removeprefix("[SPEAKER_01] ") for p in paragraphs),
            speech,
        )
        self.assertNotIn("-->", txt)

    def test_long_silence_starts_new_paragraph_with_same_speaker(self):
        data = {"segments": [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 1.0,
             "text": "First thought."},
            {"speaker": "SPEAKER_01", "start": 7.0, "end": 8.0,
             "text": "Next topic."},
        ]}
        txt = write_txt(data, self.media, self.dest).read_text()
        self.assertEqual(txt.count("[SPEAKER_01]"), 2)

    def test_txt_no_grouping(self):
        text = write_txt(self.data, self.media, self.dest,
                         group_turns=False).read_text()
        self.assertEqual(text.count("[SPEAKER_00]"), 2)

    def test_srt_constraints_and_no_lost_words(self):
        srt = write_srt(self.data, self.media, self.dest).read_text()
        blocks = srt.strip().split("\n\n")
        self.assertGreater(len(blocks), 4)
        actual = []
        for index, block in enumerate(blocks, 1):
            lines = block.splitlines()
            self.assertEqual(lines[0], str(index))
            self.assertRegex(lines[1], r"^\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}$")
            self.assertLessEqual(len(lines[2:]), 2)
            self.assertTrue(all(len(line) <= 42 for line in lines[2:]))
            cue = " ".join(lines[2:])
            for prefix in ("SPEAKER_00: ", "SPEAKER_01: ", "UNKNOWN: "):
                cue = cue.replace(prefix, "")
            actual.append(cue)
        self.assertEqual(" ".join(" ".join(actual).split()),
                         " ".join(s["text"] for s in self.data["segments"]))

    def test_no_speaker_labels_and_one_line(self):
        srt = write_srt(self.data, self.media, self.dest,
                        include_speakers=False, chars_per_line=32,
                        max_lines=1, max_duration=4.0).read_text()
        self.assertNotIn("SPEAKER", srt)
        for block in srt.strip().split("\n\n"):
            lines = block.splitlines()
            self.assertEqual(len(lines), 3)
            self.assertLessEqual(len(lines[-1]), 32)

    def test_word_timestamp_alignment(self):
        data = {"segments": [{
            "speaker": "SPEAKER_00",
            "start": 5.0, "end": 7.0, "text": "hello world",
            "words": [{"word": "hello", "start": 5.0, "end": 5.4},
                      {"word": "world", "start": 6.5, "end": 6.9}],
        }]}
        srt = write_srt(data, self.media, self.dest,
                        include_speakers=False, chars_per_line=32,
                        max_lines=1, max_duration=3.0).read_text()
        self.assertIn("00:00:05,000 --> 00:00:05,400", srt)
        self.assertIn("00:00:06,500 --> 00:00:06,900", srt)

    def test_unbroken_long_token_respects_limits(self):
        token = "https://" + "a" * 110 + ".example"
        data = {"segments": [{"speaker": "SPEAKER_00",
                              "start": 0.0, "end": 12.0, "text": token}]}
        srt = write_srt(data, self.media, self.dest,
                        include_speakers=True, chars_per_line=32,
                        max_lines=2, max_duration=4.0).read_text()
        pieces = []
        for block in srt.strip().split("\n\n"):
            lines = block.splitlines()[2:]
            self.assertLessEqual(len(lines), 2)
            for line in lines:
                self.assertLessEqual(len(line), 32)
            pieces.append("".join(line.replace("SPEAKER_00: ", "")
                                  for line in lines))
        self.assertEqual("".join(pieces), token)

    def test_missing_aligned_word_preserves_original(self):
        data = {"segments": [{
            "speaker": None, "start": 0, "end": 2, "text": "hello, world!",
            "words": [{"word": "hello,", "start": 0.0, "end": 0.4},
                      {"word": "world!"}],
        }]}
        self.assertIn("hello, world!", write_srt(
            data, self.media, self.dest, include_speakers=False).read_text())


if __name__ == "__main__":
    unittest.main()
