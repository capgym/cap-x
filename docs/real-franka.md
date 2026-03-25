# Real Franka Panda QuickStart

Make sure you have [robots_realtime](https://github.com/uynitsuj/robots_realtime.git) cloned and have tested launching the real Franka Panda using the robots_realtime repo instructions. For example test with `configs/franka/franka_robotiq_viser_teleop.yaml` (uses a robotiq gripper interfacing with an RS485).

The default task configured in [`env_configs/real/real.yaml`](../env_configs/real/real.yaml) is **"pick up the red cube and lift it"**. Feel free to modify the `task_only_prompt` and `prompt` fields in that file to define your own task, e.g. another cool task that we have tried in real (and works!) is: **"stack these objects as high as possible"**

## Requirements

Beyond a Franka Panda robot arm, this workflow requires a **stereo camera that produces calibrated metric-scale depth maps** (e.g. a ZED stereo camera). The depth data is used by SAM3 + Contact-GraspNet to generate grasp poses in 3D. A monocular RGB-only camera may be insufficient.

## Camera Extrinsics Setup

The `robots_realtime` client config points to a camera extrinsics file that describes the camera's pose in the robot world frame. You must create one for your own setup.

**1. Create your extrinsics file** in `robots_realtime/configs/camera_extrinsics/`. Use the AutoLab ZED setup as a reference:

[`configs/camera_extrinsics/autolab_franka_zed_top.yaml`](https://github.com/uynitsuj/robots_realtime/blob/main/configs/camera_extrinsics/autolab_franka_zed_top.yaml)

```yaml
# Camera extrinsics for your Franka setup.
#
# All values are expressed in the robot world frame (base link origin).
#
# position: [x, y, z] in meters
# rpy_radians: [roll, pitch, yaw] in radians (applied in that order)
#
# For your own setup, create a new file in this directory, then point
# the ZedCamera config at it via the `extrinsics_file` field.

position: [1.007, 0.0, 0.29]
rpy_radians: [1.0472, 3.14159, -1.5708]
```

**2. Point the client config at your file.** In `robots_realtime/configs/franka/franka_robotiq_client.yaml` (or whichever client config you use), update the `extrinsics_file` field under the ZedCamera sensor node:

```yaml
extrinsics_file: "configs/camera_extrinsics/your_setup.yaml"
```

**3. Calibrate your extrinsics.** The `position` and `rpy_radians` values must match your physical camera mounting. Poor calibration will cause grasp poses to be offset from the real object locations.

**4. Then run the CaP-X real experiment configs.**
```bash
uv sync --active --extra contactgraspnet
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/real/real.yaml
```

Open up the interactive web UI at the provided port (defaulted to http://localhost:8200).

Once you see:
```
Waiting for observation from real environment...
Waiting for observation from real environment...
```
In a separate terminal with the [robots_realtime](https://github.com/uynitsuj/robots_realtime.git) repo set as the current directory, launch:
```bash
uv run rr-session configs/franka/franka_robotiq_client.yaml
```
