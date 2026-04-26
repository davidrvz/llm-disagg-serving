"""
Shared runtime configuration for all workers.

Set MODEL_NAME as an environment variable to switch models without touching
code.  Both prefill and decode workers read from here so they always stay in
sync.

Examples
--------
MODEL_NAME=gpt2 python -m workers.prefill.worker          # default
MODEL_NAME=google/gemma-3-1b-it python -m workers.prefill.worker
"""

from __future__ import annotations

import os

import torch

# Any HuggingFace AutoModelForCausalLM repo ID works here.
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt2")


def select_device() -> str:
    """Return the best available device: MPS on Mac, otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
