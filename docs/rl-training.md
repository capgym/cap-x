# CaP-RL: Reinforcement Learning for Code-as-Policy Agents

CaP-RL enables on-policy reinforcement learning with verifiable environment rewards, training coding agents to generate better robot manipulation programs. We use Group Relative Policy Optimization (GRPO) to post-train language models on CaP-Gym tasks.

> **Key result:** The RL post-trained agent retains **84% success on cube lifting** and **76% on cube stacking** when deployed zero-shot to a real Franka Panda robot — no additional fine-tuning required.

---

## Table of Contents

- [Setup](#setup)
- [Training](#training)
  - [What the training script does](#what-the-training-script-does)
  - [Configuration](#configuration)
- [Evaluation](#evaluation)
  - [Merging Checkpoints](#1-merge-fsdp-checkpoints)
  - [Serving via vLLM](#2-serve-via-vllm)
  - [Running Evaluation](#3-run-evaluation)
- [Available Tasks](#available-tasks)

---

## Setup

CaP-RL requires a **separate virtual environment** from the main project because flash-attn wheels require Python 3.12.

```bash
# Create and activate a dedicated RL environment (Python 3.12 required for flash-attn)
uv venv .venv-rl --python 3.12
source .venv-rl/bin/activate

# Install RL + simulation dependencies (--active targets the activated venv)
uv sync --active --extra verl --extra robosuite

# Log in to Weights & Biases (training metrics dashboard)
wandb login

# Log in to Hugging Face (model checkpoint downloads)
huggingface-cli login
```

**Requirements:**
- Python 3.12 (flash-attn wheels require cp312)
- CUDA-capable GPU (tested on H100, A100, RTX 4090)
- [Weights & Biases](https://wandb.ai/) account for experiment tracking
- [Hugging Face](https://huggingface.co/) account for model access

---

## Training

Each task is trained independently using GRPO. Set `MODEL_PATH` to a Hugging Face model, `DATA_ROOT` for output, and `DATA_SOURCE` for the environment name.

### Cube Lift

```bash
source .venv-rl/bin/activate
MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
DATA_ROOT=output/franka_cube_lift \
DATA_SOURCE=franka_lift_code_env \
bash scripts/train_franka_grpo.sh
```

### Cube Stack

```bash
source .venv-rl/bin/activate
MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
DATA_ROOT=output/franka_cube_stack \
DATA_SOURCE=franka_robosuite_pick_place_code_env \
bash scripts/train_franka_grpo.sh
```

### Spill Wipe

```bash
source .venv-rl/bin/activate
MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
DATA_ROOT=output/franka_robosuite_spill_wipe_code_env \
DATA_SOURCE=franka_robosuite_spill_wipe_code_env \
bash scripts/train_franka_grpo.sh
```

Training logs and checkpoints are saved to `DATA_ROOT`. Monitor training progress on your W&B dashboard.

### Configuration

Override these key environment variables for training on different tasks and models:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HuggingFace model ID or local path |
| `DATA_ROOT` | `$HOME/data/capx/franka` | Output directory for dataset and checkpoints |
| `DATA_SOURCE` | `franka_pick_place_code_env` | Environment name (see [Available Tasks](#available-tasks)) |
| `N_GPUS` | auto-detected | Number of GPUs to use |
| `GROUP_SIZE` | `15` | GRPO group size (rollouts per prompt) |
| `TRAIN_TEMPERATURE` | `1.0` | Sampling temperature during training |
| `TRAIN_DATASET_SIZE` | `256` | Number of training prompts |
| `VAL_DATASET_SIZE` | `256` | Number of validation prompts |
| `PYROKI_PORT` | `8116` | Port for the PyRoKi IK server |

---

## Evaluation

Evaluation requires three steps: merge distributed checkpoints, serve the model, and run the evaluation harness.

### 1. Merge FSDP Checkpoints

After training, GRPO saves FSDP-sharded checkpoints. Merge them into a standard Hugging Face format:

```bash
# Checkpoints are saved under DATA_ROOT/checkpoints/capx/<experiment_name>/
export CKPT_PATH=<path-to-global_step_N/actor>
export TARGET_DIR=${CKPT_PATH%/actor}/hf_checkpoint

source .venv-rl/bin/activate
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CKPT_PATH \
    --target_dir $TARGET_DIR
```

### 2. Serve via vLLM

Launch the merged checkpoint as an OpenAI-compatible API server:

```bash
source .venv-rl/bin/activate
python -m capx.serving.vllm_server --model $TARGET_DIR
```

The server starts on `http://localhost:8000` by default.

### 3. Run Evaluation

Point the CaP-X evaluation harness at the served model. The `--model` flag must match the path passed to vLLM in step 2:

```bash
source .venv-rl/bin/activate
python capx/envs/launch.py \
    --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml \
    --server-url http://127.0.0.1:8000/chat/completions \
    --model $TARGET_DIR \
    --total-trials 100
```

---

## Available Tasks

| Task | `DATA_SOURCE` | Config | Description |
|------|---------------|--------|-------------|
| Cube Lift | `franka_lift_code_env` | `cube_lifting/` | Lift a red cube above a height threshold |
| Cube Stack | `franka_robosuite_pick_place_code_env` | `cube_stack/` | Stack red cube on green cube |
| Spill Wipe | `franka_robosuite_spill_wipe_code_env` | `spill_wipe/` | Wipe a spill with a sponge |

See `env_configs/` for more tasks and the full list of YAML configurations. 
