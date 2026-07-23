from pathlib import Path


def test_dataset_is_generated_before_oracle_audit() -> None:
    source = Path("scripts/train_franka_grpo.sh").read_text()

    generation = source.index("python -m capx.cli.prepare_verl_dataset")
    audit = source.index("oracle programs are forbidden in policy-training data")
    trainer = source.index("python -m verl.trainer.main_ppo")

    assert generation < audit < trainer
