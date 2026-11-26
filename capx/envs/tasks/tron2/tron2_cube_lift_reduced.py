"""TRON2 Cube Lift with RGB-D, segmentation, grasp, IK, and joint primitives."""

from capx.envs.tasks.base import CodeExecutionEnvBase


class Tron2CubeLiftReducedCodeEnv(CodeExecutionEnvBase):
    """Evaluate programs that compose primitive operations to lift a cube."""

    prompt = """
You are controlling the left arm of a LimX TRON2.
Pick up the red cube and hold it at least 10 cm above the table.

Use the available perception, inverse kinematics, joint motion, and gripper
primitives. Positions for the robot are meters in base_Link (+X forward,
+Y left, +Z up). Orientations are quaternions in WXYZ order. Joint angles
are radians. Read motion return values before commanding another movement.

The listed API functions are available in the execution environment.
Write executable Python only. Import numpy if you need it.
"""
