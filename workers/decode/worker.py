"""
Decode worker: receives a KV cache from a prefill worker and runs the
autoregressive decode loop to completion.  No tokenization of the prompt
happens here — the prompt was already processed by the prefill worker.

This is the latency-sensitive phase: one forward pass per output token.
Model is controlled by the MODEL_NAME env var (see workers/config.py).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from kv_transfer.serializer import deserialize_kv_cache
from workers.config import MODEL_NAME, select_device

logger = logging.getLogger(__name__)


class DecodeWorker:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or select_device()
        self.model: PreTrainedModel | None = None
        self.tokenizer = None

    def load(self, model_name: str = MODEL_NAME) -> None:
        """Load model weights for decoding.

        TODO: Share weights with the prefill worker when co-located (same
              process or same node) to avoid loading them twice.
        TODO: Quantize to INT8/INT4 with bitsandbytes for larger models.
        """
        logger.info("Loading %s on %s …", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Decode worker ready.")

    def decode_stream(
        self,
        kv_cache_b64: str,
        input_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Generator[str, None, None]:
        """Autoregressive decode loop; yields one decoded token string at a time.

        The prefill worker already ran the prompt through the model, so each
        step here only processes a single new token — past_key_values carries
        all prior context, keeping each forward pass O(1) in sequence length.

        Parameters
        ----------
        kv_cache_b64:
            Serialized KV cache produced by the prefill worker.
        input_ids:
            Prompt token IDs used to seed the first decode step.
        max_new_tokens:
            Hard cap on output length.
        temperature:
            Sampling temperature. 0 or below → greedy argmax.

        Yields
        ------
        str
            Decoded text for each new token (may be empty for special tokens
            consumed by skip_special_tokens=True).

        TODO: Add top-p / top-k sampling.
        TODO: Support batched decoding across multiple requests.
        """
        assert self.model is not None, "Call .load() first"
        assert self.tokenizer is not None

        past_key_values = deserialize_kv_cache(kv_cache_b64, device=self.device)
        ids = torch.tensor([input_ids], device=self.device)
        eos_id = self.tokenizer.eos_token_id

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # Only pass the last token — past_key_values supplies all prior context.
                output = self.model(
                    input_ids=ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                logits = output.logits[:, -1, :]

                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_token = int(torch.argmax(logits, dim=-1).item())

                if eos_id is not None and next_token == eos_id:
                    break

                ids = torch.cat(
                    [ids, torch.tensor([[next_token]], device=self.device)], dim=1
                )
                yield self.tokenizer.decode([next_token], skip_special_tokens=True)

    def decode(
        self,
        kv_cache_b64: str,
        input_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> tuple[str, int]:
        """Non-streaming decode — collects decode_stream() into a single string.

        Returns
        -------
        text : str
            Full decoded output (prompt excluded).
        generated_tokens : int
            Number of tokens produced.
        """
        tokens = list(
            self.decode_stream(kv_cache_b64, input_ids, max_new_tokens, temperature)
        )
        return "".join(tokens), len(tokens)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, I am a language model"

    # Prefill and decode run in the same process here just for the smoke test.
    # In production they are separate worker processes.
    from workers.prefill.worker import PrefillWorker

    print(f"\nModel  : {MODEL_NAME}")
    print(f"Prompt : {prompt!r}\n")

    prefill = PrefillWorker()
    prefill.load()
    kv_b64, input_ids, n_tokens, _ = prefill.prefill(prompt)
    print(f"Prefill done — {n_tokens} prompt tokens, {len(kv_b64)} KV bytes\n")

    decode = DecodeWorker()
    decode.load()

    print("Streaming decode:\n")
    sys.stdout.write("  ")
    token_count = 0
    for token in decode.decode_stream(kv_b64, input_ids, max_new_tokens=40, temperature=1.0):
        sys.stdout.write(token)
        sys.stdout.flush()
        token_count += 1
    print(f"\n\nGenerated {token_count} tokens")
    print("\nSmoke test PASSED")
