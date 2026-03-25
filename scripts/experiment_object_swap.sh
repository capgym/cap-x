#!/bin/bash
# Run 10 trials on libero_object_swap task 0 (alphabet soup -> basket)
# with the improved prompt config.
set -e

cd /home/gear/HyRL-main-regression

# Activate the LIBERO venv
source .venv-libero/bin/activate

# Check servers
echo "=== Checking servers ==="
all_up=true
for port in 8110 8114 8115 8116; do
    if curl -sf --connect-timeout 2 http://127.0.0.1:$port/ > /dev/null 2>&1 || \
       curl -sf --connect-timeout 2 http://127.0.0.1:$port/health > /dev/null 2>&1; then
        echo "  Port $port: UP"
    else
        echo "  Port $port: DOWN"
        all_up=false
    fi
done

if [ "$all_up" = false ]; then
    echo ""
    echo "ERROR: Some servers are not reachable."
    echo "Start servers first: bash scripts/start_servers_and_eval.sh"
    exit 1
fi

echo ""
echo "=== Running libero_object_swap task 0 (10 trials) ==="
echo "Config: franka_libero_pick_place_vdm_reduced_skill_library.yaml"
echo "Output: ./outputs/experiment_object_swap_improved/"
echo ""

CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m capx.envs.scripts.run_libero_batch \
    --args.suites libero_object_swap \
    --args.total-trials 10 \
    --args.num-workers 1 \
    --args.record-video False \
    --args.output-dir ./outputs/experiment_object_swap_improved

echo ""
echo "=== Results ==="
total=$(find outputs/experiment_object_swap_improved -name "trial_*" -type d 2>/dev/null | wc -l)
success=$(find outputs/experiment_object_swap_improved -name "*taskcompleted_1*" -type d 2>/dev/null | wc -l)
echo "Total: $total  Success: $success  Rate: $(python3 -c "print(f'{${success}/${total}:.2f}' if ${total} > 0 else 'N/A')")"
