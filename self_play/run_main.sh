#!/bin/bash

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
ABBREVIATION="qwen2.5-1.5b-instruct-tool-selfplay"
ITERATIONS=5
TRAIN_STEPS_GEN=50
TRAIN_STEPS_SOLVER=50
BASE_DIR="${SELF_PLAY_OUTPUT_DIR:-./${ABBREVIATION}}"
SOLVER_PORT="${SOLVER_PORT:-5000}"
export SOLVER_BASE_URL="http://127.0.0.1:${SOLVER_PORT}/v1"
VLLM_LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$VLLM_LOG_DIR"
export VLLM_SERVER_PID_FILE="${VLLM_LOG_DIR}/vllm_server_${SLURM_JOB_ID:-$$}.pid"
export VLLM_SERVER_LOG="${VLLM_LOG_DIR}/vllm_solver_v0.log"
VRAM_LOG_INTERVAL_SECONDS="${VRAM_LOG_INTERVAL_SECONDS:-60}"
VRAM_MONITOR_PID=""
if [[ ! "$VRAM_LOG_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "VRAM_LOG_INTERVAL_SECONDS must be a positive integer" >&2
    exit 1
fi
if command -v nvidia-smi >/dev/null; then
    (
        while true; do
            printf '[%s] GPU VRAM (index, used MiB, total MiB):\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
            sleep "$VRAM_LOG_INTERVAL_SECONDS"
        done
    ) &
    VRAM_MONITOR_PID=$!
    echo "Logging GPU VRAM every ${VRAM_LOG_INTERVAL_SECONDS}s to the job output"
else
    echo "nvidia-smi is unavailable; skipping VRAM logging" >&2
fi

stop_solver_server() {
    local server_pid
    if [[ ! -f "$VLLM_SERVER_PID_FILE" ]]; then
        return 0
    fi
    read -r server_pid < "$VLLM_SERVER_PID_FILE"
    if [[ ! "$server_pid" =~ ^[0-9]+$ ]]; then
        echo "Invalid vLLM PID in $VLLM_SERVER_PID_FILE" >&2
        return 1
    fi
    kill "$server_pid" 2>/dev/null || true
    for ((attempt=1; attempt<=30; attempt++)); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            if ! python -c 'import sys, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + sys.argv[1] + "/v1/models", timeout=1).close()' "$SOLVER_PORT" >/dev/null 2>&1; then
                rm -f "$VLLM_SERVER_PID_FILE"
                return 0
            fi
        fi
        sleep 1
    done
    echo "vLLM process $server_pid did not stop" >&2
    return 1
}
cleanup() {
    if [[ -n "$VRAM_MONITOR_PID" ]]; then
        kill "$VRAM_MONITOR_PID" 2>/dev/null || true
        wait "$VRAM_MONITOR_PID" 2>/dev/null || true
    fi
    stop_solver_server
}
trap cleanup EXIT

export WANDB_DISABLED=false
export WANDB_PROJECT="self-play-$ABBREVIATION"

echo "----------------------------------------------------------------"
echo "Starting Self-Play (Iter 1 explicit + Loop for 2..$ITERATIONS)"
echo "Base Model: $BASE_MODEL"
echo "----------------------------------------------------------------"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-$BASE_DIR}/triton_autotune}"
mkdir -p "$TRITON_CACHE_DIR"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export VLLM_USE_V1=0
export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_ASYNC_ERROR_HANDLING=1


echo ">>> STARTING ITERATION 1 (Initialization)"

GEN_V1_DIR="${BASE_DIR}/iter1_generator"
SOLVER_V1_DIR="${BASE_DIR}/iter1_solver"
DATA_V1_JSON="${BASE_DIR}/iter1_data.json"

echo "[Iter 1] Step 0: Hosting Base Model (Solver v0)..."
bash run_step0.sh "$BASE_MODEL" "$SOLVER_PORT" || exit 1

echo "[Iter 1] Step 1: Training Generator v1..."
bash run_step1.sh \
    "$BASE_MODEL" \
    "$GEN_V1_DIR" \
    "${ABBREVIATION}-iter1-gen" \
    "$TRAIN_STEPS_GEN" || exit 1

GEN_V1_CKPT="${GEN_V1_DIR}/merged-${TRAIN_STEPS_GEN}"

stop_solver_server || exit 1


export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "[Iter 1] Step 2: Generating Data..."
bash run_step2.sh \
    "$GEN_V1_CKPT" \
    "$BASE_MODEL" \
    "$DATA_V1_JSON" || exit 1

echo "[Iter 1] Step 3: Training Solver v1..."
bash run_step3.sh \
    "$BASE_MODEL" \
    "$DATA_V1_JSON" \
    "$SOLVER_V1_DIR" \
    "${ABBREVIATION}-iter1-solver" \
    "$TRAIN_STEPS_SOLVER" || exit 1

echo ">>> FINISHED ITERATION 1"


for (( i=2; i<=ITERATIONS; i++ ))
do
    prev=$((i-1))
    echo ">>> STARTING ITERATION $i (Derived from v$prev)"

    PREV_SOLVER_DIR="${BASE_DIR}/iter${prev}_solver"
    PREV_SOLVER_CKPT="${PREV_SOLVER_DIR}/merged-${TRAIN_STEPS_SOLVER}"
    PREV_GEN_DIR="${BASE_DIR}/iter${prev}_generator"
    PREV_GEN_CKPT="${PREV_GEN_DIR}/merged-${TRAIN_STEPS_GEN}"


    CURR_GEN_DIR="${BASE_DIR}/iter${i}_generator"
    CURR_SOLVER_DIR="${BASE_DIR}/iter${i}_solver"
    CURR_DATA_JSON="${BASE_DIR}/iter${i}_data.json"

    if [ ! -d "$PREV_SOLVER_CKPT" ]; then
        echo "ERROR: Previous solver checkpoint not found at $PREV_SOLVER_CKPT"
        exit 1
    fi
    if [ ! -d "$PREV_GEN_CKPT" ]; then
        echo "ERROR: Previous generator checkpoint not found at $PREV_GEN_CKPT"
        exit 1
    fi


    echo "[Iter $i] Step 0: Hosting Solver v$prev..."
    export VLLM_SERVER_LOG="${VLLM_LOG_DIR}/vllm_solver_v${prev}.log"
    bash run_step0.sh "$PREV_SOLVER_CKPT" "$SOLVER_PORT" || exit 1


    echo "[Iter $i] Step 1: Training Generator v$i..."
    bash run_step1.sh \
        "$PREV_GEN_CKPT" \
        "$CURR_GEN_DIR" \
        "${ABBREVIATION}-iter${i}-gen" \
        "$TRAIN_STEPS_GEN" || exit 1


    CURR_GEN_CKPT="${CURR_GEN_DIR}/merged-${TRAIN_STEPS_GEN}"

    stop_solver_server || exit 1

    echo "[Iter $i] Step 2: Generating Data..."
    bash run_step2.sh \
        "$CURR_GEN_CKPT" \
        "$PREV_SOLVER_CKPT" \
        "$CURR_DATA_JSON" || exit 1

    echo "[Iter $i] Step 3: Training Solver v$i..."
    bash run_step3.sh \
        "$PREV_SOLVER_CKPT" \
        "$CURR_DATA_JSON" \
        "$CURR_SOLVER_DIR" \
        "${ABBREVIATION}-iter${i}-solver" \
        "$TRAIN_STEPS_SOLVER" || exit 1

    echo ">>> FINISHED ITERATION $i"
done

echo "================================================================"
echo "Self-Play Complete."
echo "Model: ${BASE_DIR}/iter${ITERATIONS}_solver/merged-${TRAIN_STEPS_SOLVER}"
echo "================================================================"
