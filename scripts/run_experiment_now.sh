#!/bin/bash
# Quick experiment: run 5 trials per task on libero_object_swap with improved prompt
# Usage: bash scripts/run_experiment_now.sh
set -e
cd "$(git rev-parse --show-toplevel)"

echo "=== Server Check ==="
for port in 8110 8114 8115 8116; do
    if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/$port" 2>/dev/null; then
        echo "  Port $port: UP"
    else
        echo "  Port $port: DOWN -- please start servers first"
        echo "  Run: bash scripts/start_servers_and_eval.sh"
        exit 1
    fi
done

echo ""
echo "=== Launching Experiment ==="
echo "Suite: libero_object_swap (10 tasks × 5 trials = 50 trials)"
echo "Config: improved prompt (goto_pose, Molmo-first, flat code)"
echo "Workers: 2"
echo ""

rm -rf outputs/experiment_improved 2>/dev/null

source .venv-libero/bin/activate
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m capx.envs.scripts.run_libero_batch \
    --args.suites libero_object_swap \
    --args.num-workers 2 \
    --args.total-trials 5 \
    --args.output-dir ./outputs/experiment_improved \
    2>&1 | tee logs/experiment_improved.log &

BATCH_PID=$!
echo "Batch PID: $BATCH_PID"
echo ""
echo "Monitor:"
echo "  tail -f logs/experiment_improved.log"
echo "  watch -n 30 'find outputs/experiment_improved -name \"trial_*\" -type d | wc -l'"
echo "  watch -n 30 'find outputs/experiment_improved -name \"*taskcompleted_1*\" -type d | wc -l'"
