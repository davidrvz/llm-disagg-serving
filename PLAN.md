# llm-disagg-serving — project plan

## what we're building
A disaggregated LLM inference stack built from scratch, for learning and demo purposes.
Disaggregated serving splits the two phases of LLM inference onto separate workers:
- **prefill** — processes the input prompt, produces KV cache (compute-bound)
- **decode** — generates tokens one at a time, consumes KV cache (memory-bandwidth-bound)

Separating them lets you scale each independently and eliminates GPU contention between phases.

## stack
- **model**: GPT-2 via HuggingFace transformers (CPU/MPS on Mac to start)
- **workers**: FastAPI + uvicorn, one process per role
- **router**: FastAPI gateway that schedules requests across workers
- **kv transfer**: serialized numpy tensors over HTTP (local), abstracted for future network transport
- **ui**: Next.js 14 (app router), streaming via SSE, live pipeline visualization
- **infra**: docker-compose for local multi-process, Makefile for convenience

## phases

### phase 1 — local MVP (current)
- [ ] scaffold complete (done)
- [ ] PLAN.md in place (done)
- [ ] implement prefill worker: tokenize prompt → forward pass → extract + serialize KV cache
- [ ] implement decode worker: receive KV cache → autoregressive decode → stream tokens
- [ ] implement router: accept request → call prefill → pass KV to decode → stream response
- [ ] wire up with docker-compose, verify end-to-end on Mac
- [ ] basic Next.js UI: chat input, streaming token output

### phase 2 — true disagg + observability
- [ ] prefill and decode as fully separate processes (already true) — measure the KV transfer cost
- [ ] per-request latency breakdown: route time, prefill time, KV transfer time, decode time
- [ ] UI pipeline view: live stage indicators, TTFT, tok/s, KV size metrics
- [ ] worker utilization bars in UI
- [ ] experiments/latency_breakdown.ipynb — sweep prompt lengths, measure each phase

### phase 3 — cloud (optional, costs money)
- [ ] deploy prefill worker to GPU instance (A10 or L4 on Lambda Labs / RunPod)
- [ ] deploy decode worker separately
- [ ] measure real network KV transfer cost vs collocated baseline
- [ ] swap GPT-2 for a larger model (Llama 3.2 1B or 3B)

## session log

### session 1
- scaffolded full repo structure (router, workers, kv_transfer, ui, infra, experiments)
- created GitHub repo, pushed initial scaffold
- stubs in place with docstrings and TODOs throughout
- created PLAN.md

## current focus
Phase 1 — implement the prefill worker (workers/prefill/worker.py)

## key decisions / notes
- using GPT-2 to start: small enough to run on CPU, real transformer architecture, easy to swap out
- KV cache transport is abstracted behind kv_transfer/transport.py so we can swap local → network later without touching worker code
- UI is designed around real-time pipeline visibility, not just chat — the point is to *see* disagg serving happening
- keep the router dumb for now (round-robin), scheduling strategy is phase 2+
