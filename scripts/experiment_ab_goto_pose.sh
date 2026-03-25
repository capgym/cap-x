#!/usr/bin/env bash
# A/B test: goto_pose (Config B) vs solve_ik+move_to_joints (Config A)
# Suite: libero_object_swap, 10 trials each
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG_A="env_configs/libero_pick_place/experiment_A_no_goto.yaml"
CONFIG_B="env_configs/libero_pick_place/franka_libero_pick_place_vdm_reduced_skill_library.yaml"
SUITE="libero_object_swap"
TRIALS=10
NUM_WORKERS=4

OUTPUT_A="outputs/experiment_A_no_goto"
OUTPUT_B="outputs/experiment_B_goto_pose"

echo "========================================================================"
echo "A/B Test: goto_pose vs solve_ik+move_to_joints"
echo "Suite: $SUITE | Trials per task: $TRIALS"
echo "========================================================================"

# ── Experiment A: solve_ik + move_to_joints (OLD approach) ──────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Config A (solve_ik + move_to_joints)..."
echo "Config: $CONFIG_A"
echo "Output: $OUTPUT_A"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

uv run --no-sync --active python -m capx.envs.scripts.run_libero_batch \
    --base-config-path "$CONFIG_A" \
    --suites "$SUITE" \
    --output-dir "$OUTPUT_A" \
    --total-trials "$TRIALS" \
    --num-workers "$NUM_WORKERS" \
    --record-video True

echo ""
echo "Config A complete."

# ── Experiment B: goto_pose (NEW approach) ──────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Config B (goto_pose)..."
echo "Config: $CONFIG_B"
echo "Output: $OUTPUT_B"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

uv run --no-sync --active python -m capx.envs.scripts.run_libero_batch \
    --base-config-path "$CONFIG_B" \
    --suites "$SUITE" \
    --output-dir "$OUTPUT_B" \
    --total-trials "$TRIALS" \
    --num-workers "$NUM_WORKERS" \
    --record-video True

echo ""
echo "Config B complete."

# ── Compare results ─────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "RESULTS COMPARISON"
echo "========================================================================"
echo ""

print_results() {
    local label="$1"
    local output_dir="$2"

    echo "── $label ──"
    # Find all result JSON files and compute aggregate stats
    local total_tasks=0
    local total_successes=0
    local total_trials_count=0

    for task_dir in "$output_dir/$SUITE"/*/; do
        [ -d "$task_dir" ] || continue
        task_name="$(basename "$task_dir")"

        # Look for result files in model subdirectories
        for run_dir in "$task_dir"*/run/*/; do
            [ -d "$run_dir" ] || continue
            successes=0
            trials_found=0
            for result_file in "$run_dir"*/result.json 2>/dev/null; do
                [ -f "$result_file" ] || continue
                trials_found=$((trials_found + 1))
                # Check if success
                if python3 -c "import json; r=json.load(open('$result_file')); exit(0 if r.get('success', False) or r.get('reward', 0) >= 1.0 else 1)" 2>/dev/null; then
                    successes=$((successes + 1))
                fi
            done
            if [ "$trials_found" -gt 0 ]; then
                rate=$(python3 -c "print(f'{$successes/$trials_found:.1%}')")
                echo "  $task_name: $successes/$trials_found ($rate)"
                total_successes=$((total_successes + successes))
                total_trials_count=$((total_trials_count + trials_found))
                total_tasks=$((total_tasks + 1))
            fi
        done
    done

    if [ "$total_trials_count" -gt 0 ]; then
        overall=$(python3 -c "print(f'{$total_successes/$total_trials_count:.1%}')")
        echo "  ─────────────────────────────────"
        echo "  OVERALL: $total_successes/$total_trials_count ($overall) across $total_tasks tasks"
    else
        echo "  (no results found)"
    fi
    echo ""
}

print_results "Config A: solve_ik + move_to_joints (OLD)" "$OUTPUT_A"
print_results "Config B: goto_pose (NEW)" "$OUTPUT_B"

echo "========================================================================"
echo "A/B test complete."
echo "========================================================================"
