#!/usr/bin/env bash
set -euo pipefail

export DATA_SOURCE=${DATA_SOURCE:-franka_nut_assembly_code_env}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}
export DATA_ROOT=${DATA_ROOT:-$HOME/data/capx/nut_assembly_prompt_only}
export GROUP_SIZE=${GROUP_SIZE:-8}
export TRAIN_DATASET_SIZE=${TRAIN_DATASET_SIZE:-64}
export VAL_DATASET_SIZE=${VAL_DATASET_SIZE:-64}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
export PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-4}
export ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-4}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
export SAVE_FREQ=${SAVE_FREQ:-1}
export TEST_FREQ=${TEST_FREQ:-1}
export TRAINER_LOGGER=${TRAINER_LOGGER:-console}

exec bash "$(dirname "$0")/train_franka_grpo.sh"
