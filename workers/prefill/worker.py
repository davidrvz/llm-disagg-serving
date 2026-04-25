"""
Prefill worker: tokenize the prompt and run a single forward pass to populate
the KV cache.  The resulting cache is serialized and handed off to a decode
worker — the prefill GPU never touches autoregressive decoding.

Model: GPT-2 (runs on CPU/MPS, easy to swap for any CausalLM).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from kv_transfer.serializer import serialize_kv_cache

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt2"


class PrefillWorker:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model: GPT2LMHeadModel | None = None
        self.tokenizer = None

    def load(self) -> None:
        """Download (or load from cache) GPT-2 weights and move to device.

        TODO: Accept model_name as a config param so this swaps to any
              HuggingFace CausalLM (Llama, Mistral, …) without code changes.
        TODO: Use bfloat16 on CUDA/MPS to cut memory in half.
        """
        logger.info("Loading %s on %s …", MODEL_NAME, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()
        logger.info("Model ready.")

    @torch.inference_mode()
    def prefill(self, prompt: str) -> tuple[str, list[int], int]:
        """Tokenize *prompt* and run one forward pass to build the KV cache.

        Returns
        -------
        kv_cache_b64 : str
            Base64-encoded serialized KV cache (see kv_transfer.serializer).
        input_ids : list[int]
            Token IDs for the prompt (needed by the decode worker to continue).
        prompt_tokens : int
            Number of prompt tokens.

        TODO: Implement proper past_key_values extraction from model output.
        TODO: Support chunked prefill for very long prompts.
        TODO: Pin the KV tensor to shared memory (kv_transfer.transport) when
              the decode worker is co-located to skip serialization entirely.
        """
        assert self.model is not None, "Call .load() first"
        assert self.tokenizer is not None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids: list[int] = inputs["input_ids"][0].tolist()

        # Forward pass — use_cache=True populates past_key_values
        output: CausalLMOutputWithCrossAttentions = self.model(
            **inputs,
            use_cache=True,
        )

        # past_key_values is a tuple of (key, value) tensors per layer
        past_key_values = output.past_key_values  # type: ignore[assignment]

        # TODO: Replace stub serialization with real implementation
        kv_cache_b64 = serialize_kv_cache(past_key_values)

        return kv_cache_b64, input_ids, len(input_ids)
