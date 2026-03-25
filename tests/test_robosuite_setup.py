# Set environment variable for EGL rendering
import os

os.environ["MUJOCO_GL"] = "egl"

import imageio
import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="NutAssemblySquare",  # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,  # Disable on-screen rendering to avoid GLFW/X11 issues
    has_offscreen_renderer=True,
    use_camera_obs=True,
)

# reset the environment
env.reset()

# Video recording setup
frames = []
video_path = "robosuite_demo.mp4"

for _ in range(100):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    # Remove env.render() call to avoid GLFW/X11 display issues

    # Capture frame for video
    frame = env.sim.render(width=640, height=480, camera_name="birdview")
    frames.append(frame[::-1])  # flip vertically for correct orientation

# Save video
print(f"Saving video to {video_path}")
imageio.mimsave(video_path, frames, fps=30)
print("Video saved successfully!")
