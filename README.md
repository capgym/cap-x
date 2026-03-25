


# CaP-X

### A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation

[Project Page](https://capgym.github.io/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2603.22435) 

**Max Fu<sup>&#42;,1,2</sup>, Justin Yu<sup>&#42;,2</sup>, Karim El-Refai<sup>&#42;,2</sup>, Ethan Kou<sup>&#42;,2</sup>, Haoru Xue<sup>&#42;,1,2</sup>,
Huang Huang<sup>3</sup>, Wenli Xiao<sup>4</sup>, Guanzhi Wang<sup>1</sup>, Fei-Fei Li<sup>3</sup>, Guanya Shi<sup>4</sup>, Jiajun Wu<sup>3</sup>,
Shankar Sastry<sup>2</sup>, Yuke Zhu<sup>1</sup>, Ken Goldberg<sup>&dagger;,2</sup>, Jim Fan<sup>&dagger;,1</sup>**

<sup>1</sup>NVIDIA &ensp; <sup>2</sup>UC Berkeley &ensp; <sup>3</sup>Stanford University &ensp; <sup>4</sup>Carnegie Mellon University

<sup>&#42;</sup>Equal contribution &ensp; <sup>&dagger;</sup>Equal advising



---
**CaP-X** is an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. It consists of four components:


| Component      | What it does                                                                                                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CaP-Gym**    | Interactive Gymnasium environments where agents control robots by generating Python code that composes perception and control primitives. 39 tasks across Robosuite, LIBERO-PRO, and BEHAVIOR. |
| **CaP-Bench**  | Systematic benchmark evaluating coding agents across abstraction levels, interaction modes, and visual grounding modalities. 8 tiers (S1-S4 single-turn, M1-M4 multi-turn).                    |
| **CaP-Agent0** | Training-free agentic framework with multi-turn visual differencing, auto-synthesized skill libraries, and parallel ensembled reasoning.                                                       |
| **CaP-RL**     | Reinforcement learning on the coding agent via GRPO, using environment rewards to post-train language models. Transfers from sim to real with minimal gap.                                     |


---

## Installation

CaP-X uses [uv](https://docs.astral.sh/uv/) for dependency management. Requires **Python 3.10** and a **CUDA-capable GPU**.

```bash
git clone --recurse-submodules https://github.com/capgym/cap-x && cd CaP-X

# Or if already cloned without --recurse-submodules:
git submodule update --init --recursive

# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.10 && uv venv -p 3.10

# Base install
uv sync
```

### Simulator-specific setup

Pick **one** simulator family to install — Robosuite and LIBERO conflict in the same environment.

#### Robosuite

```bash
uv sync --extra robosuite
```

#### LIBERO-PRO

LIBERO requires a **separate virtual environment** (its Robosuite fork conflicts with the standard one).

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

See [docs/libero-tasks.md](docs/libero-tasks.md) for running any of 130+ LIBERO tasks.

#### BEHAVIOR (Isaac Sim)

BEHAVIOR tasks run on NVIDIA Isaac Sim via OmniGibson. Requires Python 3.10 and CUDA 12.x.

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset          # installs OmniGibson, Isaac Sim, BDDL, cuRobo, and downloads assets
cd ../../..                        # back to repo root

# Post-install fixes — copy cuRobo JIT headers to site-packages
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

> The `--dataset` flag downloads robot assets, BEHAVIOR-1K scene/object assets, and 2025 challenge task instances. You will be prompted to accept the NVIDIA Isaac Sim EULA and BEHAVIOR dataset license. To auto-accept, add `--accept-dataset-tos`.

For headless servers:
```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
# Remove duplicate Vulkan ICD if present (causes segfault on multi-GPU systems)
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
```

See [docs/behavior-tasks.md](docs/behavior-tasks.md) for task details and expected baselines.

### Optional extras

```bash
uv sync --extra verl             # RL training with VeRL/GRPO
uv sync --extra contactgraspnet  # Contact-GraspNet grasp planning
uv sync --extra curobo           # cuRobo GPU-accelerated IK & motion planning (requires CUDA)
```

## Quick Start

### 1. Perception servers (auto-launched)

Perception servers (SAM3, ContactGraspNet, PyRoKi) are **auto-launched** by the YAML config when you run an evaluation. No manual setup required for most configs.

To pre-launch servers (e.g. for sharing across multiple eval runs):

```bash
# Start SAM3 + GraspNet + PyRoKi with automatic GPU allocation
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

Use `--dry-run` to preview the allocation. Other profiles:

```bash
--profile full      # All perception servers (SAM3, GraspNet, PyRoKi, OWL-ViT, SAM2)
--profile minimal   # PyRoKi only (for oracle/privileged evals)
```

### 2. Set up an LLM proxy

The evaluation harness queries an LLM through a local proxy that exposes an OpenAI-compatible API.

```bash
# OpenRouter (get a key at openrouter.ai/keys)
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

> **Note:** `.openrouterkey` are git-ignored. The default server URL in configs is `http://127.0.0.1:8110/chat/completions`. 

See [docs/configuration.md](docs/configuration.md) for all provider options (OpenRouter, NVIDIA, vLLM, custom).

### 3. Run evaluation

```bash
# Robosuite: single-turn benchmark (100 trials, 12 parallel workers)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "google/gemini-3.1-pro-preview"

# Robosuite: multi-turn with visual differencing
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
    --model "google/gemini-3.1-pro-preview"

# LIBERO-PRO: spatial task (requires .venv-libero)
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model "google/gemini-3.1-pro-preview"

# BEHAVIOR: R1Pro radio pickup (20 trials)
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "google/gemini-3.1-pro-preview"

# Interactive Web UI
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --web-ui True
# Open http://localhost:8200

# Regression tests
./scripts/regression_test.sh quick    # 10-trial smoke test (~30s)
./scripts/regression_test.sh test1    # Full single-turn (~3 min)
```

> **Tip (BEHAVIOR):** Isaac Sim uses `OMNIGIBSON_GPU_ID` (not `CUDA_VISIBLE_DEVICES`) to select the GPU. For best performance on multi-GPU systems, run perception servers on a separate GPU (e.g. `OMNIGIBSON_GPU_ID=0` for the eval, and pre-launch SAM3/GraspNet with `CUDA_VISIBLE_DEVICES=1`). Set `OMNI_KIT_ACCEPT_EULA=YES` and `OMNIGIBSON_HEADLESS=1` for headless evaluation.

---

## Documentation

| Guide | Contents |
| ----- | -------- |
| [Adding Environments](docs/adding-environments.md) | Creating simulators, task environments, YAML configs |
| [Adding APIs](docs/adding-apis.md) | Implementing and registering new robot control APIs |
| [Configuration](docs/configuration.md) | YAML format, CLI flags, LLM provider setup |
| [LIBERO-PRO Tasks](docs/libero-tasks.md) | Setup, running any of 130+ LIBERO tasks, suite reference |
| [BEHAVIOR Tasks](docs/behavior-tasks.md) | Setup, R1Pro tasks, expected baselines, environment variables |
| [Development](docs/development.md) | Testing, linting, LIBERO/GraspNet setup, checkpoints, known issues |
| [Real-World Franka Panda Bringup](docs/real-franka.md) | Bringup with robots_realtime, real-robot QuickStart |
| [RL Training](docs/rl-training.md) | CaP-RL with GRPO/VeRL, sim-to-real transfer |

---

## Citation

```bibtex
@inproceedings{fu2025capx,
  title     = {{CaP-X}: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation},
  author    = {Fu, Max and Yu, Justin and El-Refai, Karim and Kou, Ethan and Xue, Haoru and Huang, Huang and Xiao, Wenli and Wang, Guanzhi and Li, Fei-Fei and Shi, Guanya and Wu, Jiajun and Sastry, Shankar and Zhu, Yuke and Goldberg, Ken and Fan, Jim},
  year      = {2025}
}
```

## License

This project is released under the [MIT License](LICENSE).
