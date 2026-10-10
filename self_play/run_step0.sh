#!/bin/bash
set -e
MODEL_PATH=${1:-"Qwen/Qwen2.5-1.5B-Instruct"}
PORT=${2:-5000}
SERVER_PID_FILE="${VLLM_SERVER_PID_FILE:-$PWD/logs/vllm_server.pid}"
SERVER_LOG="${VLLM_SERVER_LOG:-$PWD/logs/vllm_server.log}"
mkdir -p "$(dirname "$SERVER_PID_FILE")" "$(dirname "$SERVER_LOG")"

# export VLLM_DISABLE_COMPILE_CACHE=1
echo "Starting vLLM on port $PORT with model: $MODEL_PATH"

export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name solver \
  --port $PORT \
  --enforce-eager \
  --gpu-memory-utilization 0.30 \
  --tensor-parallel-size 1 > "$SERVER_LOG" 2>&1 &

server_pid=$!
echo "$server_pid" > "$SERVER_PID_FILE"
echo "Waiting for vLLM to initialize..."
for ((attempt=1; attempt<=150; attempt++)); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Check $SERVER_LOG" >&2
    exit 1
  fi
  if python -c 'import sys, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + sys.argv[1] + "/v1/models", timeout=2).close()' "$PORT" >/dev/null 2>&1; then
    echo "vLLM is ready on port $PORT"
    exit 0
  fi
  sleep 2
done

echo "Timed out waiting for vLLM. Check $SERVER_LOG" >&2
exit 1
