#!/bin/bash
# Quick test: 5 trials of LIBERO object_swap task 0
# Uses the VDM reduced skill library config with the improved prompt
# Single Gemini-3-Pro (no ensemble) to compare with old 0/18 baseline
# Requires: SAM3 (8114), GraspNet (8115), PyRoKi (8116), NV LLM (8110)
set -e

cd "$(git rev-parse --show-toplevel)"

SUITE="libero_object_swap"
TASK_ID=0
TRIALS=5
WORKERS=1
OUTPUT_DIR="./outputs/test_libero_prompt"
CONFIG_BASE="env_configs/libero_pick_place/franka_libero_pick_place_vdm_reduced_skill_library.yaml"

echo "=== Quick Prompt Test: $SUITE task $TASK_ID ==="
echo "Config: $CONFIG_BASE"
echo "Trials: $TRIALS, Workers: $WORKERS, Ensemble: OFF (single Gemini-3-Pro)"
echo ""

# --- Server health check ---
echo "=== Checking required servers ==="
MISSING=false
for port_label in "8114:SAM3" "8115:GraspNet" "8116:PyRoKi"; do
    port="${port_label%%:*}"
    label="${port_label##*:}"
    code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://127.0.0.1:$port/" 2>/dev/null || echo "000")
    if [ "$code" != "000" ]; then
        echo "  $label ($port): UP"
    else
        echo "  $label ($port): DOWN"
        MISSING=true
    fi
done

# NV LLM server
code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 3 http://127.0.0.1:8110/health 2>/dev/null || echo "000")
if [ "$code" = "200" ]; then
    echo "  NV LLM (8110): UP"
else
    echo "  NV LLM (8110): DOWN"
    MISSING=true
fi

if [ "$MISSING" = true ]; then
    echo ""
    echo "ERROR: Some servers are not running. Start them first with:"
    echo "  bash scripts/start_servers_and_eval.sh"
    echo "Or start them individually (see logs/ for output)."
    exit 1
fi

echo ""
echo "=== All servers healthy. Generating test config... ==="

# Generate a config with the specific task embedded
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
    'privileged': False,
    'max_steps': 4000,
    'seed': None,
    'enable_render': False,
    'viser_debug': False,
}
config['trials'] = $TRIALS
config['num_workers'] = $WORKERS
config['record_video'] = False
config['use_parallel_ensemble'] = False
config['use_multimodel'] = False
config['output_dir'] = '$OUTPUT_DIR/run'

with open('$TEST_CONFIG', 'w') as f:
    yaml.dump(config, f)

print(f'Config written to $TEST_CONFIG')
"

echo ""
echo "=== Launching evaluation ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Activate the libero venv and run
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
