"""Inspect installed pinned wheel; no model, audio, GPU operation or network."""
import importlib.metadata
import inspect
import unittest
from types import SimpleNamespace
from mlx_whisper import transcribe
from mlx_whisper.decoding import DecodingTask
from mlx_whisper.tokenizer import get_tokenizer
from faster_whisper.transcribe import WhisperModel
from whisperx.asr import FasterWhisperPipeline
from prompt_handling import prepare_prompt


class InstalledPromptContract(unittest.TestCase):
    def test_exact_mlx_version_and_reset(self):
        self.assertEqual(importlib.metadata.version("mlx-whisper"), "0.4.3")
        source = inspect.getsource(transcribe)
        self.assertIn('tokenizer.encode(" " + initial_prompt.strip())', source)
        self.assertIn("all_tokens[prompt_reset_since:]", source)
        self.assertIn("if not condition_on_previous_text or result.temperature > 0.5:", source)
        self.assertIn("prompt_reset_since = len(all_tokens)", source)

    def test_actual_mlx_decoder_keeps_last_223(self):
        tokenizer = get_tokenizer(True, num_languages=100, language="en", task="transcribe")
        prompt = list(range(250))
        fake = SimpleNamespace(options=SimpleNamespace(prefix=None, prompt=prompt),
                               tokenizer=tokenizer, n_ctx=448, sample_len=224,
                               sot_sequence=tokenizer.sot_sequence)
        actual = DecodingTask._get_initial_tokens(fake)
        self.assertEqual(actual, (tokenizer.sot_prev, *prompt[-223:], *tokenizer.sot_sequence))
        plan = prepare_prompt("mlx-large-v3", "A technical interview.", "Soldering, oscilloscope")
        self.assertEqual(plan["effective_tokens"],
                         len(tokenizer.encode(" " + plan["initial_prompt"])))

    def test_exact_cpu_source_contract(self):
        self.assertEqual(importlib.metadata.version("faster-whisper"), "1.2.1")
        self.assertEqual(importlib.metadata.version("whisperx"), "3.8.6")
        source = inspect.getsource(WhisperModel.get_prompt)
        self.assertIn('tokenizer.encode(" " + hotwords.strip())', source)
        self.assertIn("hotwords_tokens[: self.max_length // 2 - 1]", source)
        self.assertIn("previous_tokens[-(self.max_length // 2 - 1) :]", source)
        source = inspect.getsource(FasterWhisperPipeline)
        self.assertIn('tokenizer.encode(initial_prompt)', source)
        self.assertIn('hotwords=options.hotwords', source)


if __name__ == "__main__":
    unittest.main()
