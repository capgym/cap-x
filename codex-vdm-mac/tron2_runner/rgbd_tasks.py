"""Cube lift using measured RGB-D geometry and model-selected grasp directions."""

from capx.envs.tasks.base import CodeExecutionEnvBase


class Tron2CubeLiftRgbdCodeEnv(CodeExecutionEnvBase):
    """Evaluate a policy that chooses contact geometry and explicit TCP moves."""

    prompt = """Control the left arm of the LimX TRON2. Pick up the red cube and
leave it held at least 10 cm above its initial height. The height criterion
applies to the red cube itself.

Coordinates are in meters in base_Link. +X is forward, +Y is left, +Z is up.
The open gripper spans about 75 mm.

Call `reading = observe_object("red cube")` and check `reading["ok"]`.
`reading["bounds_midpoint_m"]` estimates the center of its visible surfaces.
`reading["visible_extent_m"]` estimates their span along the base-frame axes.
Nearby background depth can help interpret the surface but may include other objects.
Before declaring completion, call `observe_object("red cube")` again and verify
that the cube's Z increased by at least 0.10 m using comparable before-and-after
RGB-D observations with `ok=True`. The TCP height and jaw opening alone cannot
confirm a successful lift.

Choose your own desired point between the finger pads, the direction the fingers
extend, and the direction across the jaws. The two directions are XYZ vectors in
base_Link. Convert the geometry into a TCP pose with this Python interface:

    tcp_xyz, quat_wxyz = make_grasp_pose(contact_center_m, finger_direction, jaw_direction)

The converter applies the calibrated finger-pad offset and handles quaternion
conventions. Pass its two outputs to `goto_pose_left` and keep the quaternion
when moving with the same orientation. Choose the approach, contact, and lift
waypoints that suit your chosen geometry. Open the jaws before entering the grasp.
Log the chosen directions and label each motion with a descriptive `stage=`.
Use `check_pose_left(pos, quat_wxyz)` to test candidates without moving the arm.
You may query alternatives in the same code block when `plan_ok` is False.
Check the result of every actual `goto_pose_left` before continuing. If that
move is rejected or misses its target, let this block end normally so the next
turn can use the logs and the measured robot state to revise the policy.

Use the listed observation and control APIs. Return runnable Python, including
the call that runs your plan if you define a function. Import numpy if needed.
"""
