.PHONY: dev prefill decode router ui install lint fmt

COMPOSE = docker compose -f infra/docker-compose.yml

# ── Local development (no Docker) ─────────────────────────────────────────────

install:
	uv pip install -e ".[dev]"

router:
	PREFILL_URLS=http://localhost:8001 DECODE_URLS=http://localhost:8002 \
	uvicorn router.main:app --host 0.0.0.0 --port 8000 --reload

prefill:
	uvicorn workers.prefill.server:app --host 0.0.0.0 --port 8001 --reload

decode:
	uvicorn workers.decode.server:app --host 0.0.0.0 --port 8002 --reload

ui:
	cd ui && npm run dev

# ── Docker Compose ─────────────────────────────────────────────────────────────

dev:
	$(COMPOSE) up --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

# ── Code quality ───────────────────────────────────────────────────────────────

lint:
	ruff check .

fmt:
	ruff format .
