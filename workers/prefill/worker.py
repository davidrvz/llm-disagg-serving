"""
Prefill worker: tokenize the prompt and run a single forward pass to populate
the KV cache.  The resulting cache is serialized and handed off to a decode
worker — the prefill GPU never touches autoregressive decoding.

Model: GPT-2 (runs on CPU/MPS, easy to swap for any CausalLM).
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from kv_transfer.serializer import deserialize_kv_cache, serialize_kv_cache

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt2"


def select_device() -> str:
    """Return the best available device: MPS on Mac, otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PrefillWorker:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or select_device()
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
    def prefill(
        self, prompt: str
    ) -> tuple[str, list[int], int, torch.Tensor]:
        """Tokenize *prompt* and run one forward pass to build the KV cache.

        Returns
        -------
        kv_cache_b64 : str
            Base64-encoded serialized KV cache (see kv_transfer.serializer).
        input_ids : list[int]
            Token IDs for the prompt (needed by the decode worker to continue).
        prompt_tokens : int
            Number of prompt tokens.
        last_hidden_state : torch.Tensor
            Final-layer hidden states, shape (seq_len, hidden_size), on CPU.

        TODO: Support chunked prefill for very long prompts.
        TODO: Pin the KV tensor to shared memory (kv_transfer.transport) when
              the decode worker is co-located to skip serialization entirely.
        """
        assert self.model is not None, "Call .load() first"
        assert self.tokenizer is not None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids: list[int] = inputs["input_ids"][0].tolist()

        # use_cache=True  → past_key_values populated
        # output_hidden_states=True → hidden_states tuple populated (one per layer + embedding)
        output: CausalLMOutputWithCrossAttentions = self.model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
        )

        # past_key_values: tuple of (key, value) per layer
        past_key_values = output.past_key_values  # type: ignore[assignment]
        kv_cache_b64 = serialize_kv_cache(past_key_values)

        # hidden_states[-1]: final transformer layer output, shape (1, seq_len, hidden_size)
        # Squeeze batch dim and move to CPU so callers don't need to know the worker device.
        last_hidden_state: torch.Tensor = output.hidden_states[-1][0].cpu()  # type: ignore[index]

        return kv_cache_b64, input_ids, len(input_ids), last_hidden_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, I am a language model"

    worker = PrefillWorker()
    worker.load()

    kv_b64, input_ids, n_tokens, last_hidden = worker.prefill(prompt)

    # Round-trip the KV cache through deserialize to catch any serialization bugs.
    kv = deserialize_kv_cache(kv_b64, device="cpu")

    print(f"\nPrompt        : {prompt!r}")
    print(f"Device        : {worker.device}")
    print(f"Prompt tokens : {n_tokens}")
    print(f"Input IDs     : {input_ids}")
    print(f"KV layers     : {len(kv)}")
    print(f"KV[0] shapes  : key={tuple(kv[0][0].shape)}, value={tuple(kv[0][1].shape)}")
    print(f"KV b64 bytes  : {len(kv_b64)}")
    print(f"Last hidden   : shape={tuple(last_hidden.shape)}, dtype={last_hidden.dtype}")
    print("\nSmoke test PASSED")
