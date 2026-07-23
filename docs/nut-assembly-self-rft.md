# RoboSuite NutAssembly Self-RFT

This record covers the 2026-07-24 single-GPU Code-as-Policy experiment on RoboSuite
`NutAssemblySquare`. No expert or oracle program was used for generation, training, or evaluation.

## Anti-cheat gates

- Prompt parquet ground-truth programs were `None` for every row.
- Evaluation and training-source summaries had `use_oracle_code=false`.
- Generated programs could not access `env`, `APIS`, oracle symbols, filesystem or process
  introspection, undocumented imports, or exception swallowing.
- The self-RFT loader required `taskcompleted_1`, reran the policy guard, rejected oracle runs, and
  rejected an exact match to the repository oracle program.
- Formal evaluation used a clean commit and `scripts/validate_nut_assembly_results.py` checked all
  100 generated programs.

## Iteration results

| Stage | Model / configuration | Result |
| --- | --- | --- |
| Sparse-reward baseline | Qwen2.5-Coder-0.5B, LoRA-32, 20 GRPO steps | 0/20 task completion |
| Non-oracle discovery | Qwen2.5-Coder-7B-Instruct, temperature 0.8 | 7/20 task completion |
| Source robustness check | Self-generated trial 15, seeds 0-19 | 20/20 task completion |
| Self-RFT | LoRA-32, 16 optimizer steps, learning rate `2e-4` | train loss `0.04623` |
| Post-train screen | temperature 0.2, 20 trials | 20/20 task completion |
| Formal evaluation | temperature 0.2, 100 trials, 4 workers, video enabled | 100/100 task completion |

The formal run took `265.295s`, or `0.377` aggregate trials/s (`22.62` trials/min). It produced 100
successful generated programs and 200 MP4 files. The retained trial-1 combined demo is H.264,
`512x512`, 30 fps, `2.167s`, and 122,185 bytes.

## Receipts

- Experiment code commit: `adc71afc8bf4ea6b15d75a9ee569e3b6fbf8c8e2`
- Prompt parquet SHA-256: `717b705ce3a80eac33412458d5df6eb81004b2cc5bd7bca1627b564a0129438f`
- Self-generated training program SHA-256:
  `868703503a0bca98409f7530f0a7f3d573c06a131c4a194c1c6531aaa8e3ccd9`
- Reloaded merged model: 7,615,616,512 parameters, all finite, BF16
- Demo MP4 SHA-256: `b0b2ea61c536ef4946fa36c961e1c55bceeee7ce587755c0d1221ecd047aade9`

Coder A artifact roots:

```text
/home/coder/share/capx-runs/nut-assembly-qwen7-self-rft-20260724-v1
/home/coder/share/capx-runs/_home_coder_share_capx-runs_nut-assembly-qwen7-self-rft-20260724-v1_hf_checkpoint/nut-assembly-qwen7-self-rft-formal100-video-20260724
```

Validation command:

```bash
python scripts/validate_nut_assembly_results.py "$RESULT_DIR" \
  --expected-trials 100 \
  --minimum-success-rate 0.8 \
  --require-video
```
