# Adding a New API for Models to Use

APIs define the tools that LLM-generated code can call. Each API is a set of Python functions with docstrings that become the model's interface documentation.

## Step 1: Implement the API

```python
# capx/integrations/my_robot/control.py
from capx.integrations.base_api import ApiBase

class MyRobotControlApi(ApiBase):
    def __init__(self, env):
        super().__init__(env)
        # Initialize any perception/planning services

    def functions(self):
        """Return the dict of functions exposed to LLM-generated code.

        Each function's signature and docstring will be shown to the model.
        Write clear, complete docstrings — they ARE the model's documentation.
        """
        return {
            "move_to": self.move_to,
            "grasp": self.grasp,
            "get_object_position": self.get_object_position,
        }

    def move_to(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Move the robot end-effector to a target pose.

        Args:
            position: (3,) XYZ target in meters, world frame.
            orientation: (4,) WXYZ unit quaternion.

        Returns:
            None
        """
        # Implementation using self._env (the low-level simulator)
        ...

    def get_object_position(self, object_name: str) -> np.ndarray:
        """Get the 3D position of a named object.

        Args:
            object_name: Natural language name of the object (e.g., "red cube").

        Returns:
            position: (3,) XYZ in meters, world frame.
        """
        ...
```

## Step 2: Register

In `capx/integrations/__init__.py`:

```python
from .my_robot.control import MyRobotControlApi
register_api("MyRobotControlApi", MyRobotControlApi)
```

## Step 3: Reference in task configs

```yaml
apis:
  - MyRobotControlApi
```

## Design guidelines

- Function names should be self-descriptive (`get_object_pose`, not `gop`)
- Docstrings are critical — they are the *only* documentation the model sees
- Return numpy arrays with documented shapes and dtypes
- Use `self._log_step(tool_name, text, images)` for web UI execution logging
- For shared motion/vision utilities, see `capx/integrations/franka/common.py`
