# Adding a New Simulator / Task / Robot

## Step 1: Create a simulator wrapper

Subclass `BaseEnv` in `capx/envs/simulators/`. Implement the five required methods:

```python
# capx/envs/simulators/my_sim.py
from capx.envs.base import BaseEnv

class MySimEnv(BaseEnv):
    def __init__(self, max_steps=1000, privileged=False, enable_render=False):
        super().__init__()
        self.max_steps = max_steps
        # Initialize your simulator here

    def reset(self, *, seed=None, options=None):
        # Reset simulator, return (observation_dict, info_dict)
        ...

    def step(self, action):
        # Execute action, return (obs, reward, terminated, truncated, info)
        ...

    def get_observation(self):
        # Return dict with camera images, robot state, object poses
        # Required keys for vision APIs:
        #   obs["robot0_robotview"]["images"]["rgb"]   — (H, W, 3) uint8
        #   obs["robot0_robotview"]["images"]["depth"] — (H, W, 1) float32
        #   obs["robot0_robotview"]["intrinsics"]      — (3, 3) float64
        #   obs["robot0_robotview"]["pose_mat"]        — (4, 4) float64
        ...

    def compute_reward(self):
        # Return float reward for the current state
        ...

    def task_completed(self):
        # Return bool — has the task been solved?
        ...
```

Register in `capx/envs/simulators/__init__.py`:

```python
from capx.envs.base import register_env
from .my_sim import MySimEnv
register_env("my_sim_env", MySimEnv)
```

> **Tip:** For Robosuite-based environments, inherit from `RobosuiteBaseEnv` in `robosuite_base.py` to get video capture, camera processing, and motion control for free.

## Step 2: Create a task environment

Each task is a thin wrapper that defines a prompt and oracle code:

```python
# capx/envs/tasks/my_robot/my_task.py
from capx.envs.tasks.base import CodeExecutionEnvBase

class MyTaskCodeEnv(CodeExecutionEnvBase):
    """Short description of the task."""

    prompt = """
    You are controlling a robot with the API described below.
    Goal: [describe the task clearly]

    Key rules:
    - [important constraints the model should follow]
    - [coordinate frame conventions, units, etc.]

    Write ONLY executable Python code. No code fences.
    """

    oracle_code = """
    import numpy as np
    # Reference solution that achieves the task
    home_pose()
    pos, quat = sample_grasp_pose("target object")
    ...
    """
```

Register in `capx/envs/tasks/__init__.py`:

```python
from capx.envs.tasks.base import CodeExecEnvConfig, register_exec_env, register_config
from .my_robot.my_task import MyTaskCodeEnv

register_exec_env("my_task_code_env", MyTaskCodeEnv)
register_config("my_task_code_env", CodeExecEnvConfig(
    low_level="my_sim_env",     # Name registered in Step 1
    apis=["FrankaControlApi"],  # Which APIs to expose
))
```

The reason the config and environment registration is defined in the init file is because we want one unified access point to find the environment and their corresponding config file.

## Step 3: Create a YAML config

```yaml
# env_configs/my_task/my_task.yaml
env:
  _target_: capx.envs.tasks.my_robot.my_task.MyTaskCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: my_sim_env
    privileged: false
    apis:
      - FrankaControlApi

record_video: true
output_dir: ./outputs/my_task
trials: 100
num_workers: 12
```

## Step 4: Test

```bash
# Run oracle code to verify the environment works
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/my_task/my_task.yaml \
    --use-oracle-code True --total-trials 5

# Run with an LLM
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/my_task/my_task.yaml --total-trials 20
```

## Adding new B1K tasks

1. Create a new `${TASK_NAME}.yaml` under `OmniGibson/omnigibson/configs`
2. Create a new config under `env_configs/r1pro` and have `controller_cfg: ${TASK_NAME}.yaml` for low_level.
3. Run:
```bash
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/r1pro/${TASK_NAME}.yaml
```

## Listing available environments

```python
from capx.envs.tasks import list_exec_envs, list_configs
from capx.envs import list_envs
from capx.integrations import list_apis

print("Available environments: ", list_envs())
print("Available execution environments: ", list_exec_envs())
print("Available APIs: ", list_apis())
print("Available configurations: ", list_configs())
```
