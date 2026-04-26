#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

export MODEL_NAME="${MODEL_NAME:-gpt2}"
export PREFILL_URLS="${PREFILL_URLS:-http://localhost:8001}"
export DECODE_URLS="${DECODE_URLS:-http://localhost:8002}"

case "$1" in
  --prefill)
    exec uvicorn workers.prefill.server:app --host 0.0.0.0 --port 8001
    ;;
  --decode)
    exec uvicorn workers.decode.server:app --host 0.0.0.0 --port 8002
    ;;
  --router)
    exec uvicorn router.main:app --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "Usage: $0 --prefill | --decode | --router"
    exit 1
    ;;
esac
