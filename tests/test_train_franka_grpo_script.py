import re
from pathlib import Path


def test_dataset_is_generated_before_oracle_audit() -> None:
    source = Path("scripts/train_franka_grpo.sh").read_text()

    generation = source.index("python -m capx.cli.prepare_verl_dataset")
    audit = source.index("oracle programs are forbidden in policy-training data")
    trainer = source.index("python -m verl.trainer.main_ppo")

    assert generation < audit < trainer


def test_nut_assembly_minibatch_does_not_exceed_train_batch() -> None:
    source = Path("scripts/train_nut_assembly_grpo.sh").read_text()

    train_batch = int(re.search(r"TRAIN_BATCH_SIZE:-([0-9]+)", source).group(1))
    mini_batch = int(re.search(r"PPO_MINI_BATCH_SIZE:-([0-9]+)", source).group(1))

    assert mini_batch <= train_batch


def test_nut_assembly_vllm_capacity_matches_small_rollout_group() -> None:
    source = Path("scripts/train_nut_assembly_grpo.sh").read_text()

    max_num_seqs = int(re.search(r"MAX_NUM_SEQS:-([0-9]+)", source).group(1))
    group_size = int(re.search(r"GROUP_SIZE:-([0-9]+)", source).group(1))

    assert group_size <= max_num_seqs <= 32


def test_nut_assembly_reuses_one_sequential_reward_environment() -> None:
    source = Path("scripts/train_nut_assembly_grpo.sh").read_text()

    assert "ASYNC_REWARD=${ASYNC_REWARD:-false}" in source
    assert "REWARD_MANAGER=${REWARD_MANAGER:-naive}" in source
    assert "ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-1}" in source


def test_nut_assembly_default_model_fits_single_worker_memory() -> None:
    source = Path("scripts/train_nut_assembly_grpo.sh").read_text()

    assert "Qwen/Qwen2.5-Coder-0.5B-Instruct" in source


def test_experiment_name_sanitizes_model_path() -> None:
    source = Path("scripts/train_franka_grpo.sh").read_text()

    assert "MODEL_TAG=${MODEL_PATH//\\//_}" in source
    assert "trainer.experiment_name=${EXPERIMENT_NAME}" in source
