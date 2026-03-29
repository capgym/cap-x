#!/bin/bash
# Start all API servers and launch LIBERO-PRO evaluation
# Usage: bash scripts/start_servers_and_eval.sh [--workers N] [--suites SUITE1,SUITE2,...]
set -e

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

NUM_WORKERS=${1:-4}

echo "=== Cleaning up existing batch processes ==="
pkill -f run_libero_batch 2>/dev/null || true
sleep 2

echo "=== Checking API servers ==="
all_up=true
for port in 8114 8115 8116; do
    code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:$port/ 2>/dev/null || echo "000")
    if [ "$code" != "000" ]; then
        echo "  Port $port: UP (HTTP $code)"
    else
        echo "  Port $port: DOWN - starting..."
        all_up=false
    fi
done
code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:8110/health 2>/dev/null || echo "000")
if [ "$code" = "200" ]; then
    echo "  Port 8110 (NV LLM): UP"
else
    echo "  Port 8110 (NV LLM): DOWN - starting..."
    all_up=false
fi

if [ "$all_up" = false ]; then
    echo ""
    echo "=== Starting missing servers ==="
    source .venv/bin/activate

    # SAM3 server (GPU)
    if ! curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8114/ 2>/dev/null; then
        echo "Starting SAM3 on port 8114..."
        nohup python -m capx.serving.launch_sam3_server --device cuda --port 8114 --host 127.0.0.1 > logs/sam3.log 2>&1 &
    fi

    # GraspNet server (GPU)
    if ! curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8115/ 2>/dev/null; then
        echo "Starting GraspNet on port 8115..."
        nohup python -m capx.serving.launch_contact_graspnet_server --port 8115 --host 127.0.0.1 > logs/graspnet.log 2>&1 &
    fi

    # PyRoKi server (CPU)
    if ! curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8116/ 2>/dev/null; then
        echo "Starting PyRoKi on port 8116..."
        nohup python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand > logs/pyroki.log 2>&1 &
    fi

    # NV inference server
    if ! curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8110/health 2>/dev/null; then
        echo "Starting NV inference on port 8110..."
        nohup python -m capx.serving.openrouter_server --key-file .openrouterkey --port 8110 > logs/nv_server.log 2>&1 &
    fi

    echo "Waiting 30s for servers..."
    sleep 30

    # Re-check
    echo "=== Server Health Re-check ==="
    for port in 8114 8115 8116; do
        code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:$port/ 2>/dev/null || echo "000")
        echo "  Port $port: $([ "$code" != "000" ] && echo "UP" || echo "DOWN")"
    done
    code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:8110/health 2>/dev/null || echo "000")
    echo "  Port 8110: $([ "$code" = "200" ] && echo "UP" || echo "DOWN")"
fi

echo ""
echo "=== Launching LIBERO-PRO Evaluation ==="
echo "6 suites × 10 tasks × 50 trials = 3000 total trials"
echo "Workers: $NUM_WORKERS"
echo "Ensemble: Gemini-3-Pro only (3 temps) — fast mode"
echo "Output: ./outputs/libero_batch_run/"
echo ""

source .venv-libero/bin/activate
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
nohup python -m capx.envs.scripts.run_libero_batch \
    --args.num-workers $NUM_WORKERS \
    --args.output-dir ./outputs/libero_batch_run \
    > logs/libero_batch.log 2>&1 &
BATCH_PID=$!
echo "Batch PID: $BATCH_PID"
echo ""
echo "Monitor with:"
echo "  tail -f logs/libero_batch.log"
echo "  watch -n 30 'find outputs/libero_batch_run -name \"trial_*\" -type d | wc -l'"
echo "  watch -n 30 'find outputs/libero_batch_run -name \"*taskcompleted_1*\" -type d | wc -l'"
