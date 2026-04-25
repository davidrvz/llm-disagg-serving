# llm-disagg-serving

A disaggregated LLM inference system that separates **prefill** and **decode**
into independent, independently-scalable worker pools.

---

## What is disaggregated serving?

A standard LLM generation request has two fundamentally different phases:

### Prefill
The model reads your entire prompt — all tokens in parallel — and builds the
**KV cache**: a set of key/value tensors (one pair per attention layer) that
encode the context.  This phase is **compute-bound**: it does a lot of FLOPs
in one shot.

### Decode
The model generates tokens **one at a time**.  Each step takes the last token
plus the growing KV cache and produces the next token.  This phase is
**memory-bandwidth-bound**: it reads the entire KV cache every step, but does
relatively few FLOPs.

### Why separate them?

| Property | Prefill | Decode |
|---|---|---|
| Bottleneck | Compute (FLOP/s) | Memory bandwidth (GB/s) |
| Batching benefit | Huge — batch many prompts together | Modest — batches share bandwidth |
| Latency sensitivity | First-token latency | Inter-token latency |
| Ideal hardware | High-FLOP GPUs (A100, H100) | High-bandwidth GPUs or even CPUs |

When they share a machine, long prefills **block** ongoing decodes (head-of-line
blocking), increasing inter-token latency for all running requests.  Separating
them lets each pool scale independently and optimise for its own bottleneck.

This is the core idea behind systems like
[Distserve](https://arxiv.org/abs/2401.09670) and
[Splitwise](https://arxiv.org/abs/2311.18677).

---

## Architecture

```
Client
  │
  ▼
┌─────────────────────────────────────────┐
│  Router  (port 8000)                    │
│  FastAPI gateway + round-robin scheduler│
└────────────┬──────────────┬────────────┘
             │              │
    ┌────────▼───┐    ┌─────▼──────┐
    │  Prefill   │    │   Decode   │
    │  Worker    │    │   Worker   │
    │  (8001)    │    │   (8002)   │
    └────────────┘    └────────────┘
             │              ▲
             └──── KV ──────┘
               (serialized,
                HTTP for now)
```

1. **Router** receives `POST /generate`, picks a prefill and a decode worker
   via round-robin scheduling.
2. **Prefill worker** tokenizes the prompt, runs a single forward pass with
   `use_cache=True`, serializes `past_key_values` to base64, and returns it.
3. **Decode worker** deserializes the KV cache and runs the autoregressive loop
   until EOS or `max_new_tokens`.
4. Router returns the assembled response to the client.

---

## Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- Node.js 20+ (for the UI)

### 1. Install dependencies

```bash
cd llm-disagg-serving
cp .env.example .env
uv pip install -e ".[dev]"
```

### 2. Start workers (three terminals)

```bash
# Terminal 1 — prefill worker
make prefill

# Terminal 2 — decode worker
make decode

# Terminal 3 — router
make router
```

### 3. Send a request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The history of artificial intelligence", "max_new_tokens": 64}'
```

### 4. Open the UI

```bash
make ui   # starts Next.js on http://localhost:3000
```

### 5. Run with Docker Compose

```bash
make dev   # builds and starts router + prefill + decode + ui
```

### 6. Run the latency notebook

```bash
cd experiments
jupyter lab latency_breakdown.ipynb
```

---

## Project layout

```
llm-disagg-serving/
├── router/               # FastAPI gateway + scheduler + Pydantic models
├── workers/
│   ├── prefill/          # Tokenize → forward pass → KV cache
│   └── decode/           # KV cache → autoregressive decode loop
├── kv_transfer/          # Serialization + transport abstractions
├── ui/                   # Next.js 14 playground
├── experiments/          # Jupyter notebooks
├── infra/                # Docker Compose + Makefile + Dockerfiles
└── pyproject.toml
```

---

## Key TODOs (roughly in priority order)

- [ ] `workers/prefill/worker.py` — verify `past_key_values` extraction is correct
- [ ] `workers/decode/worker.py` — test the incremental decode loop end-to-end
- [ ] `kv_transfer/transport.py` — implement `SharedMemoryTransport` to skip HTTP for co-located workers
- [ ] `router/main.py` — add SSE streaming (`req.stream=True`)
- [ ] `router/scheduler.py` — replace round-robin with load-aware scheduling
- [ ] `infra/docker-compose.yml` — add GPU device passthrough for CUDA workers
- [ ] Scale to a larger model (GPT-2 medium → Llama 3.2 1B)
