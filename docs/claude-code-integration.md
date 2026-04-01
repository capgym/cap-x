# CaP-X + Claude Code: Autonomous Research Agent

Claude Code can be used as an autonomous research agent that iteratively improves robot manipulation success rates. Instead of controlling the robot directly, Claude Code runs evaluations through the standard `launch.py` harness, analyzes results, modifies prompts and skills, and re-evaluates — forming a closed-loop hillclimbing system.

## Setup

### 1. Install CaP-X and a simulator (see README)

```bash
git clone --recurse-submodules https://github.com/capgym/cap-x && cd CaP-X
uv sync --extra robosuite   # or --extra libero, etc.
```

### 2. Start perception servers

Perception servers are auto-launched by YAML configs, or start them manually:

```bash
# PyRoKi (CPU, port 8116)
uv run --no-sync --active python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1

# SAM3 (GPU, port 8114)
CUDA_VISIBLE_DEVICES=1 uv run --no-sync --active python -m capx.serving.launch_sam3_server --port 8114 --device cuda --host 127.0.0.1

# ContactGraspNet (GPU, port 8115)
CUDA_VISIBLE_DEVICES=2 uv run --no-sync --active python -m capx.serving.launch_contact_graspnet_server --port 8115 --host 127.0.0.1
```

### 3. Start LLM proxy

```bash
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

### 4. Open Claude Code

```bash
cd CaP-X && claude
```

The `.claude/commands/` slash commands and `CLAUDE.md` project instructions are auto-discovered.

## The Autoresearch Loop

The core workflow is a hillclimbing loop where Claude Code autonomously:

### Step 1: Establish baseline

```
> /run-eval env_configs/cube_stack/franka_robosuite_cube_stack.yaml --total-trials 20
```

### Step 2: Analyze failures

Claude reads trial outputs to identify failure modes:

```bash
# Examine failed trials
find outputs/<model>/<task>/ -name "*taskcompleted_0*" -type d
cat outputs/<model>/<task>/trial_XX_.../summary.txt
```

Common failure patterns: missed grasps, wrong placement height, collision, timeout.

### Step 3: Improve prompts and code

Claude modifies the task prompt, oracle code, or API implementation:

```python
# Task prompts live in:
capx/envs/tasks/franka/franka_pick_place.py   # prompt + oracle code
env_configs/cube_stack/*.yaml                   # YAML config overrides

# API implementations:
capx/integrations/franka/control.py            # Robosuite control API
capx/integrations/franka/libero.py             # LIBERO visual API
capx/integrations/r1pro/control.py             # BEHAVIOR R1Pro API
```

### Step 4: Evolve the skill library

Extract reusable functions from successful trials:

```
> /skill-library extract
> /skill-library show
```

Skills are stored in `.capx_skills.json` and auto-injected into future prompts.

### Step 5: Re-evaluate

```
> /run-eval env_configs/cube_stack/franka_robosuite_cube_stack.yaml --total-trials 100
```

Compare against baseline and iterate.

## Parallel Evaluation

For large-scale sweeps across multiple GPUs:

```bash
# Run different tasks on different GPUs
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --total-trials 100 --num-workers 12 &

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml \
    --total-trials 100 --num-workers 12 &
```

## Available Slash Commands

| Command | Purpose |
|---------|---------|
| `/run-eval <config> [args]` | Run benchmark evaluation |
| `/skill-library [show\|add\|extract]` | Manage the evolving skill library |
| `/regression-test [quick\|test1\|test2]` | Run regression tests |

## Key Files for Experimentation

| File | Purpose |
|------|---------|
| `capx/envs/tasks/franka/*.py` | Task prompts and oracle code |
| `capx/integrations/franka/*.py` | Control APIs (what the LLM-generated code can call) |
| `capx/envs/trial.py` | Multi-turn execution loop (VDM, regeneration) |
| `capx/skills/library.py` | Skill library (extraction, promotion, injection) |
| `capx/llm/client.py` | LLM query routing and ensemble config |
| `.capx_skills.json` | Persisted skill library |
| `env_configs/` | YAML configs (model, temperature, trials, servers) |

## What Claude Code Can Modify

- Prompts in YAML configs and task `.py` files
- Skills in `.capx_skills.json`
- Model selection, temperature, multi-turn settings (CLI flags or YAML)
- API implementations in `capx/integrations/`
- Number of workers, trials, retry logic

## What Should Not Be Modified

- Simulator internals (`capx/third_party/`)
- Reward functions (defined by the simulator)
- Test-time information assumptions
