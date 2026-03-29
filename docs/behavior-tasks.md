# BEHAVIOR Tasks

BEHAVIOR tasks use the [BEHAVIOR-1K](https://behavior.stanford.edu/) benchmark via NVIDIA Isaac Sim and OmniGibson. The R1Pro humanoid robot performs everyday household manipulation tasks in physically simulated environments.

## Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 12.x
- Isaac Sim 4.5.0 (installed via `uv_install.sh`)

## Installation

From the repo root:

```bash
# Clone the BEHAVIOR-1K submodule (hosted separately)

cd capx/third_party/b1k
./uv_install.sh --dataset
cd ../../..
```

This single command installs OmniGibson, Isaac Sim, BDDL, cuRobo, downloads robot assets, BEHAVIOR-1K scene/object assets, and 2025 challenge task instances.

To auto-accept all licenses (non-interactive):

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset --accept-dataset-tos
cd ../../..
```

### Post-install fixes

After `uv_install.sh`, a few additional steps may be needed depending on your system:

```bash
# 1. Fix duplicate Vulkan ICD (causes segfault on multi-GPU systems)
#    If both /etc/vulkan/icd.d/nvidia_icd.json and /usr/share/vulkan/icd.d/nvidia_icd.json exist,
#    remove the one in /usr/share:
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json

# 2. Fix missing cuRobo CUDA headers for JIT compilation
#    The pip-installed cuRobo may be missing header files needed for first-time kernel compilation.
#    Run this with the b1k venv active:
source capx/third_party/b1k/.venv/bin/activate
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

> **Note:** On first run, cuRobo JIT-compiles CUDA kernels for the GPU architecture. This can take **3–5 minutes** and produces "JIT compiling..." warnings — this is normal.

For headless servers, also install EGL:

```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
```

## Environment variables

Set these before running BEHAVIOR tasks:

```bash
export OMNI_KIT_ACCEPT_EULA=YES        # Required: accept Isaac Sim EULA
export OMNIGIBSON_HEADLESS=1            # Required on headless servers (no display)
```

## Task overview

BEHAVIOR-1K defines **1,000+ household activities** using BDDL (BEHAVIOR Domain Definition Language). Of these, **50 tasks** form the official challenge benchmark (B10–B50 tiers). All 50 are runnable via `launch.py`.

### Running any benchmark task

All 50 benchmark tasks have `env_configs/r1pro/` YAML configs. Tasks 0 and 1 have dedicated configs with oracle code; tasks 2–49 use the generic `R1ProBehaviorCodeEnv`:

```bash
# Run any benchmark task (replace with the activity name)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/b1k_hiding_Easter_eggs.yaml

# Tasks with oracle code (original configs)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml      # turning_on_radio (oracle)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_trash.yaml       # picking_up_trash
```

### Original task configs (with oracle code)

| Config | BEHAVIOR activity | Robot | Trials | Mode |
|--------|-------------------|-------|--------|------|
| `r1pro_pick_up_radio.yaml` | `turning_on_radio` | R1Pro | 20 | Single-turn (oracle) |
| `r1pro_pick_up_radio_multiturn_vdm.yaml` | `turning_on_radio` | R1Pro | 20 | Multi-turn + VDM |
| `r1pro_pick_up_trash.yaml` | `picking_up_trash` | R1Pro | 25 | Single-turn |
| `r1pro_pick_up_trash_multiturn_vdm.yaml` | `picking_up_trash` | R1Pro | 25 | Multi-turn + VDM |
| `r1pro_pick_up_trash_oracle.yaml` | `picking_up_trash` | R1Pro | 25 | Single-turn (oracle) |

### 50 challenge benchmark tasks (B10–B50)

All 50 tasks are runnable via `launch.py`. Tasks 0–1 use dedicated configs with oracle code; tasks 2–49 use `b1k_{activity_name}.yaml` with the generic `R1ProBehaviorCodeEnv`.

**B10 (indices 0–9)**

| Index | Activity | Config |
|-------|----------|--------|
| 0 | `turning_on_radio` | `r1pro_pick_up_radio.yaml` |
| 1 | `picking_up_trash` | `r1pro_pick_up_trash.yaml` |
| 2 | `putting_away_Halloween_decorations` | `b1k_putting_away_Halloween_decorations.yaml` |
| 3 | `cleaning_up_plates_and_food` | `b1k_cleaning_up_plates_and_food.yaml` |
| 4 | `can_meat` | `b1k_can_meat.yaml` |
| 5 | `setting_mousetraps` | `b1k_setting_mousetraps.yaml` |
| 6 | `hiding_Easter_eggs` | `b1k_hiding_Easter_eggs.yaml` |
| 7 | `picking_up_toys` | `b1k_picking_up_toys.yaml` |
| 8 | `rearranging_kitchen_furniture` | `b1k_rearranging_kitchen_furniture.yaml` |
| 9 | `putting_up_Christmas_decorations_inside` | `b1k_putting_up_Christmas_decorations_inside.yaml` |

**B20 (indices 10–19)**

| Index | Activity | Config |
|-------|----------|--------|
| 10 | `set_up_a_coffee_station_in_your_kitchen` | `b1k_set_up_a_coffee_station_in_your_kitchen.yaml` |
| 11 | `putting_dishes_away_after_cleaning` | `b1k_putting_dishes_away_after_cleaning.yaml` |
| 12 | `preparing_lunch_box` | `b1k_preparing_lunch_box.yaml` |
| 13 | `loading_the_car` | `b1k_loading_the_car.yaml` |
| 14 | `carrying_in_groceries` | `b1k_carrying_in_groceries.yaml` |
| 15 | `bringing_in_wood` | `b1k_bringing_in_wood.yaml` |
| 16 | `moving_boxes_to_storage` | `b1k_moving_boxes_to_storage.yaml` |
| 17 | `bringing_water` | `b1k_bringing_water.yaml` |
| 18 | `tidying_bedroom` | `b1k_tidying_bedroom.yaml` |
| 19 | `outfit_a_basic_toolbox` | `b1k_outfit_a_basic_toolbox.yaml` |

**B30 (indices 20–29)**

| Index | Activity | Config |
|-------|----------|--------|
| 20 | `sorting_vegetables` | `b1k_sorting_vegetables.yaml` |
| 21 | `collecting_childrens_toys` | `b1k_collecting_childrens_toys.yaml` |
| 22 | `putting_shoes_on_rack` | `b1k_putting_shoes_on_rack.yaml` |
| 23 | `boxing_books_up_for_storage` | `b1k_boxing_books_up_for_storage.yaml` |
| 24 | `storing_food` | `b1k_storing_food.yaml` |
| 25 | `clearing_food_from_table_into_fridge` | `b1k_clearing_food_from_table_into_fridge.yaml` |
| 26 | `assembling_gift_baskets` | `b1k_assembling_gift_baskets.yaml` |
| 27 | `sorting_household_items` | `b1k_sorting_household_items.yaml` |
| 28 | `getting_organized_for_work` | `b1k_getting_organized_for_work.yaml` |
| 29 | `clean_up_your_desk` | `b1k_clean_up_your_desk.yaml` |

**B40 (indices 30–39)**

| Index | Activity | Config |
|-------|----------|--------|
| 30 | `setting_the_fire` | `b1k_setting_the_fire.yaml` |
| 31 | `clean_boxing_gloves` | `b1k_clean_boxing_gloves.yaml` |
| 32 | `wash_a_baseball_cap` | `b1k_wash_a_baseball_cap.yaml` |
| 33 | `wash_dog_toys` | `b1k_wash_dog_toys.yaml` |
| 34 | `hanging_pictures` | `b1k_hanging_pictures.yaml` |
| 35 | `attach_a_camera_to_a_tripod` | `b1k_attach_a_camera_to_a_tripod.yaml` |
| 36 | `clean_a_patio` | `b1k_clean_a_patio.yaml` |
| 37 | `clean_a_trumpet` | `b1k_clean_a_trumpet.yaml` |
| 38 | `spraying_for_bugs` | `b1k_spraying_for_bugs.yaml` |
| 39 | `spraying_fruit_trees` | `b1k_spraying_fruit_trees.yaml` |

**B50 (indices 40–49)**

| Index | Activity | Config |
|-------|----------|--------|
| 40 | `make_microwave_popcorn` | `b1k_make_microwave_popcorn.yaml` |
| 41 | `cook_cabbage` | `b1k_cook_cabbage.yaml` |
| 42 | `chop_an_onion` | `b1k_chop_an_onion.yaml` |
| 43 | `slicing_vegetables` | `b1k_slicing_vegetables.yaml` |
| 44 | `chopping_wood` | `b1k_chopping_wood.yaml` |
| 45 | `cook_hot_dogs` | `b1k_cook_hot_dogs.yaml` |
| 46 | `cook_bacon` | `b1k_cook_bacon.yaml` |
| 47 | `freeze_pies` | `b1k_freeze_pies.yaml` |
| 48 | `canning_food` | `b1k_canning_food.yaml` |
| 49 | `make_pizza` | `b1k_make_pizza.yaml` |

## Running evaluations

All BEHAVIOR evaluations must run with the b1k venv active and environment variables set:

```bash
source capx/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1  # only needed on headless servers

# Radio pickup (oracle, single-turn)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml

# Trash pickup (oracle, single-turn)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_trash.yaml

# Radio pickup (multi-turn with visual differencing)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio_multiturn_vdm.yaml
```

## Architecture

BEHAVIOR tasks use the following components:

- **Simulator**: `capx.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel` — wraps OmniGibson with Isaac Sim physics
- **Task envs**: `capx.envs.tasks.r1pro/` — task-specific code execution environments
- **Control API**: `capx.integrations.r1pro.control.R1ProControlApi` — perception (SAM3, ContactGraspNet) and control (cuRobo IK, action primitives)
- **OmniGibson configs**: `capx/third_party/b1k/OmniGibson/omnigibson/configs/r1pro_*.yaml`

## API servers

BEHAVIOR tasks require perception servers running alongside the simulator. The YAML configs auto-launch them, but you can also start them manually. **All BEHAVIOR commands must run with the b1k venv active:**

```bash
source capx/third_party/b1k/.venv/bin/activate

# SAM3 segmentation server
uv run --no-sync --active python -m capx.serving.launch_sam3_server --device cuda --port 8114

# ContactGraspNet grasp planning server
uv run --no-sync --active python -m capx.serving.launch_contact_graspnet_server --port 8115
```

## Troubleshooting
- If possible, run servers and behavior on different GPUs.
- **SAM3 "returned no results" on most trials** — If SAM3 perception fails frequently, check that the SAM3 server is NOT filtering low-confidence detections. The server should return ALL detections (the control API applies its own threshold). On multi-GPU systems, consider running SAM3 on a separate GPU from Isaac Sim (e.g., `device: cuda:1` in the YAML config) to avoid memory contention.
- **`ModuleNotFoundError: No module named 'omnigibson'`** — Run `./uv_install.sh` from `capx/third_party/b1k/`.
- **`ModuleNotFoundError: No module named 'isaacsim'`** — Isaac Sim wheels not installed. Re-run `./uv_install.sh`.
- **`OMNI_KIT_ACCEPT_EULA` error** — Set `export OMNI_KIT_ACCEPT_EULA=YES` before running.
- **Segfault in XR extension on headless servers** — Set `export OMNIGIBSON_HEADLESS=1`. If still crashing, check for duplicate Vulkan ICDs: `ls /etc/vulkan/icd.d/ /usr/share/vulkan/icd.d/` — if `nvidia_icd.json` exists in both, remove the one in `/usr/share/vulkan/icd.d/`.
- **cuRobo JIT compilation fails with `fatal error: helper_math.h`** — Copy missing headers: `cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h .venv/lib/python3.10/site-packages/curobo/curobolib/cpp/`
- **Rendering errors on headless servers** — Install `libegl1 libgl1` and ensure GPU is accessible.
- **Websockets conflict** — The install script auto-fixes this. If you see websockets errors, manually remove `pip_prebundle/websockets` directories under Isaac Sim's `extscache/`.
- **First run is slow** — cuRobo JIT-compiles CUDA kernels on first import (~5 min). Subsequent runs use cached kernels. Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability (e.g., `8.9` for L40/Ada) to speed up compilation.
- **PyTorch CUDA version mismatch** — OmniGibson may install PyTorch built for CUDA 13.0, which requires driver 570+. If your driver is older (e.g., 550), `uv_install.sh` auto-downgrades to `cu124`. If you see `CUDA error: no kernel image is available`, manually run: `uv pip install "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" --extra-index-url https://download.pytorch.org/whl/cu124` inside the b1k venv.
