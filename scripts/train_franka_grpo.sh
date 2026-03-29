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
  data.train_batch_size=256 \
  data.val_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
  actor_rollout_ref.actor.kl_loss_coef=0.02 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=${GROUP_SIZE} \
  actor_rollout_ref.rollout.agent.num_workers=25 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path=verl_agent_reward/capx_franka_reward.py \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger=[wandb] \
  trainer.project_name=capx \
  trainer.experiment_name=${ALGO}_${MODEL_PATH}_${DATA_SOURCE}_${DATE}_refactor_temperature_${TRAIN_TEMPERATURE}_group_size_${GROUP_SIZE} \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.default_local_dir=${DATA_ROOT}/checkpoints/\${trainer.project_name}/\${trainer.experiment_name} \
  trainer.test_freq=-1 \
  trainer.total_epochs=50 \
  trainer.val_before_train=False \
  reward_model.launch_reward_fn_async=True \
  reward_model.reward_manager=prime

  # +ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=egl \