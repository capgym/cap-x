---
description: Run a CaP-X evaluation benchmark on a simulation task.
allowed-tools: Bash, Read, Glob
argument-hint: <config-path> [--total-trials N] [--model MODEL]
---

# Run CaP-X Evaluation: $ARGUMENTS

Run an evaluation benchmark using the CaP-X launcher.

## Command

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path $ARGUMENTS
```

If no config path is provided, list available configs:
```bash
find env_configs/ -name "*.yaml" -not -path "*/hillclimb/*" | sort
```

## Available config categories

| Directory | Task |
|-----------|------|
| `cube_stack/` | Stack red cube on green cube |
| `cube_lifting/` | Lift red cube |
| `cube_restack/` | Restack cubes in new order |
| `nut_assembly/` | Peg insertion / nut assembly |
| `spill_wipe/` | Wipe spill with sponge |
| `two_arm_handover/` | Bimanual object handover |
| `two_arm_lift/` | Bimanual cooperative lift |
| `libero_pick_place/` | LIBERO benchmark tasks |
| `r1pro/` | R1Pro mobile manipulation |
| `real/` | Real Franka Panda |

## After completion

Report the summary statistics: success rate, average reward, task completed count, and elapsed time.
