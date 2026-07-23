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
