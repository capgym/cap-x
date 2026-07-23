from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from capx.envs.tasks.franka.franka_nut_assembly import ORACLE_CODE
from capx.security import validate_generated_program


@dataclass(frozen=True)
class Args:
    base_model: str
    prompt_parquet: Path
    trial_dirs: list[Path]
    output_dir: Path
    epochs: int
    repeats: int
    learning_rate: float
    lora_rank: int
    gradient_accumulation_steps: int
    max_length: int
    seed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_trial_dir(trial_dir: Path) -> dict[str, str]:
    if "taskcompleted_1" not in trial_dir.name:
        raise ValueError(f"training source is not a successful trial: {trial_dir}")
    summary_path = trial_dir.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["use_oracle_code"]:
        raise ValueError(f"oracle rollout cannot be used for self-RFT: {trial_dir}")

    code_path = trial_dir / "code.py"
    code = code_path.read_text()
    guard = validate_generated_program(code)
    if not guard.allowed:
        raise ValueError(f"training source failed policy guard: {guard.reason}")
    if code.strip() == ORACLE_CODE.strip():
        raise ValueError("expert oracle code cannot be used for self-RFT")
    return {
        "trial_dir": str(trial_dir.resolve()),
        "program_sha256": _sha256(code_path),
        "program": code,
    }


def _load_prompt_rows(prompt_parquet: Path) -> list[dict[str, str]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(prompt_parquet, columns=["prompt", "reward_model"]).to_pylist()
    if not rows:
        raise ValueError(f"prompt dataset is empty: {prompt_parquet}")
    leaked = [row for row in rows if row["reward_model"]["ground_truth"]["program"]]
    if leaked:
        raise ValueError("prompt dataset contains expert programs")
    prompt = rows[0]["prompt"]
    if any(row["prompt"] != prompt for row in rows[1:]):
        raise ValueError("self-RFT expects one task prompt repeated across seeds")
    return prompt


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Self-RFT on successful non-oracle NutAssembly policy rollouts."
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--prompt-parquet", type=Path, required=True)
    parser.add_argument("--trial-dir", dest="trial_dirs", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    namespace = parser.parse_args()
    return Args(**vars(namespace))


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _train(args: Args) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    prompt = _load_prompt_rows(args.prompt_parquet)
    sources = [_validate_trial_dir(path) for path in args.trial_dirs]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples: list[dict[str, torch.Tensor]] = []
    prompt_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
    )
    for _ in range(args.repeats):
        for source in sources:
            response_ids = tokenizer(
                source["program"] + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            input_ids = prompt_ids + response_ids
            if len(input_ids) > args.max_length:
                raise ValueError(
                    f"training example has {len(input_ids)} tokens, limit is {args.max_length}"
                )
            labels = [-100] * len(prompt_ids) + response_ids
            examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                }
            )

    class RolloutDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return examples[index]

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [item["input_ids"] for item in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "labels": pad_sequence(
                [item["labels"] for item in batch], batch_first=True, padding_value=-100
            ),
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
            ),
        }

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
            data_seed=args.seed,
        ),
        train_dataset=RolloutDataset(),
        data_collator=collate,
    )
    train_result = trainer.train()
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)

    merged_dir = args.output_dir / "hf_checkpoint"
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    model_files = sorted(merged_dir.glob("*.safetensors"))
    receipt = {
        "git_commit": _git_commit(),
        "oracle_code_used": False,
        "base_model": args.base_model,
        "prompt_parquet": str(args.prompt_parquet.resolve()),
        "prompt_parquet_sha256": _sha256(args.prompt_parquet),
        "sources": [{key: value for key, value in source.items() if key != "program"} for source in sources],
        "training_args": asdict(args),
        "train_metrics": train_result.metrics,
        "merged_model_files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in model_files
        ],
    }
    receipt["training_args"]["prompt_parquet"] = str(args.prompt_parquet)
    receipt["training_args"]["trial_dirs"] = [str(path) for path in args.trial_dirs]
    receipt["training_args"]["output_dir"] = str(args.output_dir)
    (args.output_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    print(json.dumps(_train(_parse_args()), indent=2))


if __name__ == "__main__":
    main()
