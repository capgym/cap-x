from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import tyro

os.environ.setdefault("MUJOCO_GL", "egl")

from capx.envs.tasks import get_config, get_exec_env
from capx.utils.parallel_eval import run_parallel_batches


def _load_oracle_pool(path: Path | None) -> list[str]:
    if not path:
        return []
    if not path.exists():
        return []
    try:
        table = ds.dataset(str(path)).to_table()
        programs = table.column("program").to_pylist()
        out = [str(p).strip() for p in programs if isinstance(p, (str, bytes)) and str(p).strip()]
        # Deduplicate
        seen: set[str] = set()
        uniq: list[str] = []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq
    except Exception:
        return []


def _generate_rows(
    indices: list[int],
    *,
    split: str,
    data_source: str,
    seed_base: int,
) -> list[dict]:
    env = get_exec_env(data_source)(get_config(data_source))
    rows: list[dict] = []
    try:
        for idx in indices:
            env_seed = seed_base + idx
            obs = env._get_observation()
            if env.oracle_code is not None:
                ground_truth = {"program": env.oracle_code}
            else:
                ground_truth = {"program": None}
            prompt = []
            for msg in obs["full_prompt"]:
                new_msg = msg.copy()
                # Keep content as plain string for text-only models.
                # Multimodal list format [{"type": "text", ...}] breaks
                # tokenizer.apply_chat_template for text-only models.
                if isinstance(new_msg["content"], list):
                    parts = [p["text"] for p in new_msg["content"] if p.get("type") == "text"]
                    new_msg["content"] = "\n".join(parts)
                prompt.append(new_msg)
            rows.append(
                {
                    "data_source": data_source,
                    "prompt": prompt,
                    "ability": "agent",
                    "reward_model": {
                        "style": "sim_code",
                        "ground_truth": ground_truth,
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "seed": env_seed,
                    },
                }
            )
    finally:
        env.close()
    return rows


def _sample_rows(
    *,
    split: str,
    size: int,
    data_source: str,
    seed_base: int,
    num_workers: int,
) -> list[dict]:
    if size <= 0:
        return []
    trial_ids = list(range(size))
    batch_fn = functools.partial(
        _generate_rows,
        split=split,
        data_source=data_source,
        seed_base=seed_base,
    )
    rows = run_parallel_batches(
        trial_ids,
        num_workers=max(1, num_workers),
        batch_fn=batch_fn,
    )
    rows.sort(key=lambda row: row["extra_info"]["index"])
    return rows


@dataclass
class Args:
    """Create RLHF-style parquets seeded from the Franka pick-and-place env."""

    output_dir: Path
    train_size: int = 512
    val_size: int = 128
    data_source: str = "franka_pick_place_code_env"
    seed: int = 0
    num_workers: int = 1
    oracle_pool: Path | None = None


def main(args: Args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global _ORACLE_POOL
    _ORACLE_POOL = _load_oracle_pool(args.oracle_pool)
    train_rows = _sample_rows(
        split="train",
        size=args.train_size,
        data_source=args.data_source,
        seed_base=args.seed,
        num_workers=args.num_workers,
    )
    val_rows = _sample_rows(
        split="val",
        size=args.val_size,
        data_source=args.data_source,
        seed_base=args.seed + 10000,
        num_workers=args.num_workers,
    )

    pq.write_table(pa.Table.from_pylist(train_rows), args.output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val_rows), args.output_dir / "test.parquet")

    manifest = {
        "train": len(train_rows),
        "val": len(val_rows),
        "data_source": args.data_source,
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote dataset to {args.output_dir} ({len(train_rows)} train / {len(val_rows)} val)")


if __name__ == "__main__":
    main(tyro.cli(Args))