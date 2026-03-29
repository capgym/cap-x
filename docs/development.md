# Development Guide

## Testing

### Unit tests

```bash
uv run pytest tests/test_environments.py -q
```

Run a specific test:
```bash
uv run pytest tests/test_environments.py::test_franka_pick_place_code_env -q
uv run pytest tests/test_environments.py::test_franka_nut_assembly_code_env -q
```

### Oracle code testing

Check how oracle code performs on a specific environment:
```bash
uv run tests/test_environments.py --env_name YOUR_ENV_NAME
```

### Regression tests (expected rewards)

Before merging, run these environments and verify rewards match expectations:

```bash
uv run capx/envs/launch.py --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml
# Expected avg. reward: ~0.99

uv run capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml
# Expected avg. reward: ~0.90

uv run capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml
# Expected avg. reward: ~0.50

uv run capx/envs/launch.py --config-path env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml
# Expected avg. reward: ~0.15

uv run capx/envs/launch.py --config-path env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml
# Expected avg. reward: ~0.25

uv run capx/envs/launch.py --config-path env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml
# Expected avg. reward: ~0.20
```

## Linting

```bash
ruff check          # lint
ruff check --fix    # auto-fix
ruff format         # format
```

When contributing, please use ruff (automatically installed) for linting. See [ruff docs](https://docs.astral.sh/ruff/tutorial/#getting-started).

## SAM3 access

SAM3 by facebookresearch is currently not yet an accessible HuggingFace autogenerator module, so we install it as a package via third party integrations. Before using SAM 3, please request access to the checkpoints on the SAM 3 Hugging Face repo: https://github.com/facebookresearch/sam3

Once accepted, you need to be authenticated to download the checkpoints (e.g. `hf auth login` after generating an access token).

## LIBERO-PRO installation

```bash
uv sync --extra libero --extra contactgraspnet
```

For headless servers, also install EGL rendering:
```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

See [libero-tasks.md](libero-tasks.md) for the full task reference.

## BEHAVIOR installation (Isaac Sim)

BEHAVIOR tasks require NVIDIA Isaac Sim and OmniGibson. Use the provided install script:

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset --accept-dataset-tos
cd ../../..
```

This installs:
- **BDDL** (Behavior Domain Definition Language)
- **OmniGibson** (simulator, editable install)
- **Isaac Sim 4.5.0** (downloaded as pip wheels from pypi.nvidia.com)
- **cuRobo** (GPU-accelerated motion planning, from StanfordVL fork)
- **PyRoKi** (IK solver)
- **SAM3 + ContactGraspNet dependencies** (perception server runtime deps)
- **Datasets** (robot assets, BEHAVIOR-1K assets, 2025 challenge task instances)

The script also fixes the known websockets conflict with Isaac Sim extscache.

### Post-install fix: cuRobo JIT headers

After running `uv_install.sh`, copy the cuRobo CUDA JIT headers (required for first-run kernel compilation). Run with the b1k venv active:

```bash
source capx/third_party/b1k/.venv/bin/activate
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

> **Note:** On first run, cuRobo JIT-compiles CUDA kernels (3–5 min). Isaac Sim also does initial shader compilation on first run, adding another ~3 min to startup.

### Prerequisites

- Python 3.10 (Isaac Sim wheels are cp310-only)
- NVIDIA GPU with CUDA 12.x (driver 550+)
- `libegl1` and `libgl1` for headless rendering (see above)

### Environment variables

For headless (no display) servers, set before running:
```bash
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
```

See [behavior-tasks.md](behavior-tasks.md) for task configs and expected baselines.

## Vendored submodules

We vendor some upstream repos for reproducible, offline tests. Initialize submodules after cloning:

```bash
git submodule update --init --recursive
```

## Sharp bits / known issues

1. For MuJoCo, use `condim="4"` on bodies where collision matters. The default is 3, which may lead to slippage. See [MuJoCo docs](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-condim).
2. The sandbox for code execution is local and safe-ish. For stronger isolation, use Docker or nsjail.
