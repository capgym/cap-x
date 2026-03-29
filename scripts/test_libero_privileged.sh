#!/bin/bash
# Quick test: 5 trials of LIBERO object_swap task 0 (privileged baseline)
# Uses the privileged config (ground-truth state, no perception servers)
# Requires: PyRoKi (8116), NV LLM (8110)
set -e

cd "$(git rev-parse --show-toplevel)"

SUITE="libero_object_swap"
TASK_ID=0
TRIALS=5
WORKERS=1
OUTPUT_DIR="./outputs/test_libero_privileged"
CONFIG_BASE="env_configs/libero_pick_place/franka_libero_pick_place_privileged.yaml"

echo "=== Quick Privileged Test: $SUITE task $TASK_ID ==="
echo "Config: $CONFIG_BASE"
echo "Trials: $TRIALS, Workers: $WORKERS"
echo ""

# --- Server health check (privileged only needs PyRoKi + LLM) ---
echo "=== Checking required servers ==="
MISSING=false

code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://127.0.0.1:8116/" 2>/dev/null || echo "000")
if [ "$code" != "000" ]; then
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
echo "=== All servers healthy. Generating test config... ==="

mkdir -p "$OUTPUT_DIR"
TEST_CONFIG="$OUTPUT_DIR/test_config.yaml"

python3 -c "
import yaml, copy

with open('$CONFIG_BASE') as f:
    config = yaml.safe_load(f)

config['env']['cfg']['low_level'] = {
    '_target_': 'capx.envs.simulators.libero.FrankaLiberoEnv',
    'suite_name': '$SUITE',
    'task_id': $TASK_ID,
    'privileged': True,
    'max_steps': 4000,
    'seed': None,
    'enable_render': False,
    'viser_debug': False,
}
config['trials'] = $TRIALS
config['num_workers'] = $WORKERS
config['record_video'] = False
config['output_dir'] = '$OUTPUT_DIR/run'

with open('$TEST_CONFIG', 'w') as f:
    yaml.dump(config, f)

print(f'Config written to $TEST_CONFIG')
"

echo ""
echo "=== Launching evaluation ==="
echo "Output: $OUTPUT_DIR"
echo ""

source .venv-libero/bin/activate
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    uv run --no-sync --active capx/envs/launch.py \
    --config-path "$TEST_CONFIG" \
    --model "google/gemini-3.1-pro-preview" \
    --server-url "http://127.0.0.1:8110/chat/completions"

echo ""
echo "=== Done. Check results in $OUTPUT_DIR ==="
echo "Success count:"
find "$OUTPUT_DIR" -name "*taskcompleted_1*" -type d 2>/dev/null | wc -l
echo "out of $TRIALS trials"
