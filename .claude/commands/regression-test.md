---
description: Run the CaP-X regression tests to verify that code changes haven't broken evaluation performance.
allowed-tools: Bash, Read, Grep
argument-hint: [test1|test2|both]
---

# Regression Test: $ARGUMENTS

Run the CaP-X regression tests. These verify that the codebase produces results consistent with known baselines.

## Test Definitions

**Test 1 — Single-turn cube stacking (fast, ~3 min)**
- Config: `env_configs/cube_stack/franka_robosuite_cube_stack.yaml`
- Baseline: 1.000/0.421/42 (code_gen_success / avg_reward / task_completed out of 100)
- Pass criteria: task_completed >= 38 (within variance of baseline)

**Test 2 — Multi-turn with visual differencing (slow, ~15 min)**
- Config: `env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml`
- Baseline: 1.000/0.798/79
- Pass criteria: task_completed >= 70

## Commands

Run whichever test is requested (default: both).

**Test 1:**
```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --num-workers 12
```

**Test 2:**
```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
    --num-workers 8
```

## Prerequisites

These API servers must be running (in separate terminals):
```bash
CUDA_VISIBLE_DEVICES=0 uv run --active --no-sync capx/serving/launch_sam3_server.py --port 8114 --device cuda
CUDA_VISIBLE_DEVICES=0 uv run --active --no-sync capx/serving/launch_contact_graspnet_server.py --port 8115 --device cuda
uv run --active --no-sync capx/serving/launch_pyroki_server.py --port 8116
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

## After completion

Parse the summary output and report:
1. Code generation success rate (should be 1.000)
2. Average reward
3. Task completed count (and whether it meets the pass criteria)
4. Elapsed time
5. **PASS** or **FAIL** verdict
