#!/bin/bash
# Experiment: Privileged baseline on libero_goal_swap (upper bound)
# Uses ground-truth perception (no SAM3/GraspNet), only needs PyRoKi + NV LLM
# Purpose: Establish upper bound — if this gets 0%, problem is motion planning, not perception
set -e

cd "$(git rev-parse --show-toplevel)"

SUITE="libero_goal_swap"
TRIALS=10
WORKERS=2
OUTPUT_DIR="./outputs/experiment_privileged_goal"
CONFIG_BASE="env_configs/libero_pick_place/franka_libero_pick_place_privileged.yaml"
MODEL="google/gemini-3.1-pro-preview"
SERVER_URL="http://127.0.0.1:8110/chat/completions"

echo "=== Privileged Baseline: $SUITE (Upper Bound Experiment) ==="
echo "Config: $CONFIG_BASE"
echo "Trials per task: $TRIALS, Workers: $WORKERS"
echo ""

# --- Server health check (privileged only needs PyRoKi + LLM) ---
echo "=== Checking required servers ==="
MISSING=false

# PyRoKi: check port is listening (no HTTP health endpoint on /)
if ss -tln | grep -q ':8116 '; then
    echo "  PyRoKi (8116): UP"
else
    echo "  PyRoKi (8116): DOWN"
    MISSING=true
fi

code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 3 http://127.0.0.1:8110/health 2>/dev/null || echo "000")
if [ "$code" = "200" ]; then
    echo "  NV LLM (8110): UP"
else
    echo "  NV LLM (8110): DOWN"
    MISSING=true
fi

if [ "$MISSING" = true ]; then
    echo ""
    echo "ERROR: Required servers are not running."
    echo "Start PyRoKi:  nohup python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand > logs/pyroki.log 2>&1 &"
    echo "Start NV LLM:  nohup python -m capx.serving.openrouter_server --key-file .openrouterkey --port 8110 > logs/nv_server.log 2>&1 &"
    exit 1
fi

echo ""
echo "=== All servers healthy ==="

# Create a stripped config without api_servers (servers already running)
mkdir -p "$OUTPUT_DIR" logs
STRIPPED_CONFIG="$OUTPUT_DIR/privileged_no_servers.yaml"

source .venv-libero/bin/activate

python3 -c "
import yaml
with open('$CONFIG_BASE') as f:
    config = yaml.safe_load(f)
# Remove api_servers to avoid port-bind conflict with already-running servers
config.pop('api_servers', None)
with open('$STRIPPED_CONFIG', 'w') as f:
    yaml.dump(config, f)
print(f'Stripped config written to $STRIPPED_CONFIG')
"

echo ""
echo "=== Launching $SUITE with privileged config ==="
echo ""

MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    uv run --no-sync --active python -m capx.envs.scripts.run_libero_batch \
    --args.base-config-path "$STRIPPED_CONFIG" \
    --args.suites "$SUITE" \
    --args.models "$MODEL" \
    --args.server-url "$SERVER_URL" \
    --args.total-trials "$TRIALS" \
    --args.num-workers "$WORKERS" \
    --args.output-dir "$OUTPUT_DIR" \
    --args.record-video False

echo ""
echo "=========================================="
echo "=== Privileged Baseline Results ==="
echo "=========================================="
echo ""

TOTAL_TRIALS=$(find "$OUTPUT_DIR" -name "trial_*" -type d 2>/dev/null | wc -l)
SUCCESSES=$(find "$OUTPUT_DIR" -name "*taskcompleted_1*" -type d 2>/dev/null | wc -l)

echo "Total trials completed: $TOTAL_TRIALS"
echo "Successes: $SUCCESSES"
if [ "$TOTAL_TRIALS" -gt 0 ]; then
    echo "Success rate: $(echo "scale=1; $SUCCESSES * 100 / $TOTAL_TRIALS" | bc)%"
fi
echo ""
echo "Per-task breakdown:"
for task_dir in "$OUTPUT_DIR"/"$SUITE"/*/; do
    if [ -d "$task_dir" ]; then
        task_name=$(basename "$task_dir")
        task_total=$(find "$task_dir" -name "trial_*" -type d 2>/dev/null | wc -l)
        task_success=$(find "$task_dir" -name "*taskcompleted_1*" -type d 2>/dev/null | wc -l)
        echo "  $task_name: $task_success/$task_total"
    fi
done
