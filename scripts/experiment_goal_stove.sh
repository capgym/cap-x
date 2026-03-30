#!/bin/bash
# Experiment: LIBERO libero_goal_swap task 7 — "turn_on_the_stove"
# This is the EASIEST LIBERO task (no grasping needed).
# Runs 10 trials with 1 worker using launch.py directly (batch runner has no task filter).
set -e

cd "$(git rev-parse --show-toplevel)"

echo "=== Experiment: turn_on_the_stove (libero_goal_swap task 7) ==="
echo "Trials: 10, Workers: 1"
echo ""

# ── 1. Server health check ──────────────────────────────────────────────────
echo "=== Checking API servers ==="
FAIL=false
for port in 8110 8114 8115 8116; do
    if [ "$port" = "8110" ]; then
        endpoint="http://127.0.0.1:${port}/health"
    else
        endpoint="http://127.0.0.1:${port}/"
    fi
    code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 3 "$endpoint" 2>/dev/null || echo "000")
    if [ "$code" != "000" ]; then
        echo "  Port $port: UP (HTTP $code)"
    else
        echo "  Port $port: DOWN"
        FAIL=true
    fi
done

if [ "$FAIL" = true ]; then
    echo ""
    echo "ERROR: Some servers are down. Start them first:"
    echo "  bash scripts/start_servers_and_eval.sh"
    exit 1
fi
echo "All servers healthy."
echo ""

# ── 2. Create task-specific YAML config ─────────────────────────────────────
OUTPUT_DIR="./outputs/experiment_goal_stove"
mkdir -p "$OUTPUT_DIR"

CONFIG_PATH="${OUTPUT_DIR}/config_goal_stove.yaml"
cat > "$CONFIG_PATH" << 'YAML'
env:
  _target_: capx.envs.tasks.franka.franka_libero_pick_place.FrankaLiberoPickPlaceCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level:
      _target_: capx.envs.simulators.libero.FrankaLiberoEnv
      suite_name: libero_goal_swap
      task_id: 7
      privileged: false
      max_steps: 4000
      seed: null
      enable_render: false
      viser_debug: false
    privileged: false
    apis:
    - FrankaLiberoApiReducedSkillLibrary
    prompt: 'You are controlling a Franka Emika robot with API described below.

      Goal: {libero_environment_goal}

      In this robot environment, the agentview camera is looking at the robot''s workspace
      from the front. In image space the right is world frame positive Y, up is world
      frame positive Z, and backward into the camera is world frame positive X.


      CRITICAL RULES:

      1. Write flat, sequential code under 50 lines. No functions, classes, or try/except.

      2. Use Molmo as PRIMARY perception: point_prompt_molmo(rgb, "name") then segment_sam3_point_prompt(rgb,
      point).

      3. ALWAYS use top-down quaternion np.array([0.0, 1.0, 0.0, 0.0]) for orientation.

      4. Use goto_pose(position, quaternion, z_approach=0.075) for motion — it handles
      IK and approach in one call. Do NOT use solve_ik + move_to_joints separately.

      5. Use plan_grasp + select_top_down_grasp for grasp POSITION only, override
      orientation with top-down quat.

      6. Lift to Z=0.25 after grasping. Place INTO target (center Z + 0.05), not high
      above.

      7. Capture observation ONCE at start. Do NOT re-observe after picking — arm
      occludes scene.

      8. Workspace: X [0.25, 0.75], Y [-0.35, 0.35], Z [0.0, 0.45]. If depth has 3
      dims: depth = depth[:,:,0].

      9. Pick-place pattern: open_gripper -> goto_pose(grasp, quat, z_approach=0.075)
      -> close_gripper -> goto_pose(lift) -> goto_pose(place, quat, z_approach=0.05)
      -> open_gripper -> goto_pose(retreat).


      Write ONLY executable Python code (no fences). Import numpy if needed.

      The functions (APIs) below are already imported to the environment.

      '
    multi_turn_prompt: 'The following code was executed:

      ```python

      {executed_code}

      ```

      stdout: {console_stdout}

      stderr: {console_stderr}


      If IK/server timeout errors: choose FINISH (code changes won''t help).

      If perception failed: try simpler object name or segment_sam3_text_prompt as
      fallback.

      If task incomplete with no errors: re-observe and retry the failed step.


      Respond with EXACTLY ONE of:

      - ''REGENERATE'' followed by ```python...```

      - ''FINISH'' if done or if infrastructure errors.

      '
api_servers:
- _target_: capx.serving.launch_pyroki_server.main
  port: 8116
  host: 127.0.0.1
  robot: panda_description
  target_link: panda_hand
- _target_: capx.serving.launch_contact_graspnet_server.main
  port: 8115
  host: 127.0.0.1
- _target_: capx.serving.launch_sam3_server.main
  device: cuda
  port: 8114
  host: 127.0.0.1
record_video: true
evolve_skill_library: false
output_dir: ./outputs/experiment_goal_stove/run
use_img_differencing: true
use_wrist_camera: true
use_parallel_ensemble: true
use_multimodel: true
trials: 10
num_workers: 1
YAML

echo "Config written to: $CONFIG_PATH"
echo ""

# ── 3. Run the experiment ────────────────────────────────────────────────────
echo "=== Launching evaluation ==="
echo "Output: $OUTPUT_DIR"
echo ""

source .venv-libero/bin/activate
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m capx.envs.launch \
    --config-path "$CONFIG_PATH" \
    --total-trials 10 \
    --num-workers 1 \
    2>&1 | tee "${OUTPUT_DIR}/experiment.log"

# ── 4. Report results ───────────────────────────────────────────────────────
echo ""
echo "=== Results ==="

TOTAL=$(find "$OUTPUT_DIR" -name "trial_*" -type d 2>/dev/null | wc -l)
SUCCESS=$(find "$OUTPUT_DIR" -name "*taskcompleted_1*" -type d 2>/dev/null | wc -l)

echo "Task: turn_on_the_stove (libero_goal_swap task 7)"
echo "Trials completed: $TOTAL"
echo "Successes: $SUCCESS / $TOTAL"
if [ "$TOTAL" -gt 0 ]; then
    echo "Success rate: $(echo "scale=1; $SUCCESS * 100 / $TOTAL" | bc)%"
fi
