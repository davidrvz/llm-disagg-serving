#!/usr/bin/env python3
"""
End-to-end smoke test for the disaggregated inference stack.

Assumes the services are already running:
  - Router on http://localhost:8000
  - Prefill on http://localhost:8001
  - Decode on http://localhost:8002

Run with: python -m experiments.test_e2e
or directly: python experiments/test_e2e.py
"""

from __future__ import annotations

import sys
import time

import httpx

ROUTER_URL = "http://localhost:8000"
PREFILL_URL = "http://localhost:8001"
DECODE_URL = "http://localhost:8002"

TIMEOUT = 300.0  # 5 minutes for model loading


def test_health() -> None:
    """Verify all services are up."""
    print("Testing health endpoints...")
    for name, url in [
        ("router", ROUTER_URL),
        ("prefill", PREFILL_URL),
        ("decode", DECODE_URL),
    ]:
        try:
            resp = httpx.get(f"{url}/health", timeout=10)
            assert resp.status_code == 200, f"{name} returned {resp.status_code}"
            print(f"  ✓ {name} is healthy")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            raise


def test_config() -> None:
    """Read current model and worker config from router."""
    print("\nFetching router config...")
    resp = httpx.get(f"{ROUTER_URL}/config", timeout=10)
    assert resp.status_code == 200
    config = resp.json()
    print(f"  Model: {config['model_name']}")
    print(f"  Prefill workers: {config['prefill_workers']}")
    print(f"  Decode workers: {config['decode_workers']}")


def test_non_streaming() -> None:
    """Test the non-streaming generate path."""
    print("\nTesting non-streaming /generate...")
    payload = {
        "prompt": "Explain machine learning in two sentences.",
        "max_new_tokens": 60,
        "temperature": 0.7,
        "stream": False,
    }
    resp = httpx.post(f"{ROUTER_URL}/generate", json=payload, timeout=TIMEOUT)
    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
    result = resp.json()
    assert "text" in result and result["text"], "Empty response text"
    assert result["prompt_tokens"] > 0, f"Bad prompt_tokens: {result['prompt_tokens']}"
    assert result["generated_tokens"] > 0, f"Bad generated_tokens: {result['generated_tokens']}"
    print(f"  Prompt tokens: {result['prompt_tokens']}")
    print(f"  Generated tokens: {result['generated_tokens']}")
    print(f"  Output: {result['text'][:100]}...")


def test_streaming() -> None:
    """Test the streaming generate path with SSE."""
    print("\nTesting streaming /generate...")
    payload = {
        "prompt": "What is Python?",
        "max_new_tokens": 50,
        "temperature": 1.0,
        "stream": True,
    }
    with httpx.stream("POST", f"{ROUTER_URL}/generate", json=payload, timeout=TIMEOUT) as resp:
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        assert resp.headers.get("content-type") == "text/event-stream"

        token_count = 0
        full_text = ""
        print("  Streaming tokens:")
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    print(f"    [DONE]")
                elif data.startswith("["):
                    # Metadata line
                    print(f"    {data}")
                else:
                    # Token
                    full_text += data
                    token_count += 1
                    if token_count <= 3 or token_count % 10 == 0:
                        print(f"    Token {token_count}: {data!r}")

        assert token_count > 0, "No tokens generated"
        print(f"  Total tokens streamed: {token_count}")
        print(f"  Output: {full_text[:100]}...")


def main() -> int:
    print("═" * 70)
    print("LLM Disagg Serving — End-to-End Smoke Test")
    print("═" * 70)

    try:
        test_health()
        test_config()
        test_non_streaming()
        test_streaming()

        print("\n" + "═" * 70)
        print("✓ All tests PASSED")
        print("═" * 70)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
