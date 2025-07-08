#!/bin/sh
set -e

# Default values if not set by environment
: "${MODEL_NAME:=intfloat/multilingual-e5-small}"
: "${PORT:=80}"
: "${CONCURRENCY_LIMIT:=10}"
: "${LOG_LEVEL:=info}"

echo "Starting Shea with:"
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  PORT=${PORT}"
echo "  CONCURRENCY_LIMIT=${CONCURRENCY_LIMIT}"
echo "  LOG_LEVEL=${LOG_LEVEL}"

# Use exec so Uvicorn gets PID 1 for proper signal handling
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --proxy-headers \
    --limit-concurrency ${CONCURRENCY_LIMIT} \
    --log-level ${LOG_LEVEL}
