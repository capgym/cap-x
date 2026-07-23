#!/usr/bin/env bash
# trains within 1h 29m 31s
# ~13 steps to above 95% success rate
set -euo pipefail

DATE=$(date +%m%d)
echo "DATE: ${DATE}"
DATA_SOURCE=${DATA_SOURCE:-franka_pick_place_code_env}
ALGO=${ALGO:-grpo}
GROUP_SIZE=${GROUP_SIZE:-15}
TRAIN_DATASET_SIZE=${TRAIN_DATASET_SIZE:-256}
VAL_DATASET_SIZE=${VAL_DATASET_SIZE:-256}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}
DATA_ROOT=${DATA_ROOT:-$HOME/data/capx/franka}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
N_GPUS=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
PYROKI_PORT=${PYROKI_PORT:-8116}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-8}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-16}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-25}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:--1}
TRAINER_LOGGER=${TRAINER_LOGGER:-wandb}
ACTOR_LR=${ACTOR_LR:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.02}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-${LORA_RANK}}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-false}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-false}
ASYNC_REWARD=${ASYNC_REWARD:-true}
MODEL_TAG=${MODEL_PATH//\//_}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${ALGO}_${MODEL_TAG}_${DATA_SOURCE}_${DATE}_temperature_${TRAIN_TEMPERATURE}_group_${GROUP_SIZE}}

MODEL_OVERRIDES=(
  actor_rollout_ref.model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING}
  actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
  actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}
)
if (( LORA_RANK > 0 )); then
  MODEL_OVERRIDES+=(
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=all-linear
  )
fi

# ---------------------------------------------------------------------------
# Auto-start PyRoKi IK server if not already running
# ---------------------------------------------------------------------------
if ! python -c "import socket; s=socket.create_connection(('127.0.0.1', ${PYROKI_PORT}), timeout=1); s.close()" 2>/dev/null; then
  echo "Starting PyRoKi IK server on port ${PYROKI_PORT}..."
  CUDA_VISIBLE_DEVICES="" python -m capx.serving.launch_pyroki_server \
    --port "${PYROKI_PORT}" --host 127.0.0.1 &
  PYROKI_PID=$!
  # Wait up to 60s for it to become ready
  for i in $(seq 1 60); do
    if python -c "import socket; s=socket.create_connection(('127.0.0.1', ${PYROKI_PORT}), timeout=1); s.close()" 2>/dev/null; then
      echo "PyRoKi IK server ready (PID ${PYROKI_PID})"
      break
    fi
    sleep 1
  done
  trap "kill ${PYROKI_PID} 2>/dev/null || true" EXIT
fi

# ---------------------------------------------------------------------------
# Prepare dataset (only if it doesn't already exist)
# ---------------------------------------------------------------------------
echo "DATA_ROOT: ${DATA_ROOT}"
if [[ ! -d "${DATA_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT}"
  echo "Preparing dataset..."

  python -m capx.cli.prepare_verl_dataset \
    --output-dir "${DATA_ROOT}" \
    --train-size ${TRAIN_DATASET_SIZE} \
    --val-size ${VAL_DATASET_SIZE} \
    --data-source ${DATA_SOURCE}
fi

python - "${DATA_ROOT}/train.parquet" "${DATA_ROOT}/test.parquet" <<'PY'
import sys

import pyarrow.parquet as pq

for path in sys.argv[1:]:
    rows = pq.read_table(path, columns=["reward_model"]).to_pylist()
    leaked = [row for row in rows if row["reward_model"]["ground_truth"]["program"]]
    if leaked:
        raise RuntimeError(f"oracle programs are forbidden in policy-training data: {path}")
print("Verified prompt-only training data with no oracle programs")
PY

USE_KL_LOSS=false
if [[ "${ALGO}" == "grpo" ]]; then
  USE_KL_LOSS=true
fi

export MUJOCO_GL=osmesa
# export MUJOCO_EGL_DEVICE_ID=0 # is this slowing down?
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=${ALGO} \
  data.train_files=${DATA_ROOT}/train.parquet \
  data.val_files=${DATA_ROOT}/test.parquet \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.val_batch_size=${VAL_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  "${MODEL_OVERRIDES[@]}" \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=${GROUP_SIZE} \
  actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_WORKERS} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
  actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path=verl_agent_reward/capx_franka_reward.py \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger=[${TRAINER_LOGGER}] \
  trainer.project_name=capx \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.default_local_dir=${DATA_ROOT}/checkpoints/\${trainer.project_name}/\${trainer.experiment_name} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.val_before_train=False \
  reward_model.launch_reward_fn_async=${ASYNC_REWARD} \
  reward_model.reward_manager=prime

  # +ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=egl \
