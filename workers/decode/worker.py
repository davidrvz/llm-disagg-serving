"""
Decode worker: receives a KV cache from a prefill worker and runs the
autoregressive decode loop to completion.  No tokenization happens here —
the prompt was already processed by the prefill worker.

This is the latency-sensitive phase: one forward pass per output token.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from kv_transfer.serializer import deserialize_kv_cache

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt2"


class DecodeWorker:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model: GPT2LMHeadModel | None = None
        self.tokenizer = None

    def load(self) -> None:
        """Load GPT-2 weights for decoding.

        TODO: Share weights with the prefill worker when co-located (same
              process or same node) to avoid loading them twice.
        TODO: Quantize to INT8 with bitsandbytes to fit larger models.
        """
        logger.info("Loading %s on %s …", MODEL_NAME, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()
        logger.info("Decode worker ready.")

    @torch.inference_mode()
    def decode(
        self,
        kv_cache_b64: str,
        input_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> tuple[str, int]:
        """Autoregressive decode loop starting from a pre-filled KV cache.

        Parameters
        ----------
        kv_cache_b64:
            Serialized KV cache produced by the prefill worker.
        input_ids:
            Prompt token IDs (used to seed the first decode step).
        max_new_tokens:
            Hard cap on output length.
        temperature:
            Sampling temperature.  ≤0 → greedy.

        Returns
        -------
        text : str
            Decoded output string (prompt excluded).
        generated_tokens : int
            Number of tokens produced.

        TODO: Implement the actual decode loop using past_key_values.
        TODO: Add top-p / top-k sampling.
        TODO: Yield tokens incrementally for streaming (async generator).
        TODO: Implement KV cache growth — append new k/v slices each step
              rather than recomputing from scratch.
        TODO: Early stopping on EOS token.
        """
        assert self.model is not None, "Call .load() first"
        assert self.tokenizer is not None

        past_key_values = deserialize_kv_cache(kv_cache_b64, device=self.device)

        ids = torch.tensor([input_ids], device=self.device)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            # TODO: Replace with correct incremental decode step.
            # Only pass the *last* token when past_key_values is populated.
            output = self.model(
                input_ids=ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values  # type: ignore[assignment]
            logits = output.logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = int(torch.argmax(logits, dim=-1).item())

            generated.append(int(next_token))
            ids = torch.cat([ids, torch.tensor([[next_token]], device=self.device)], dim=1)

            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and next_token == eos_id:
                break

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, len(generated)
