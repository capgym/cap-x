# uv run python tests/test_environments.py # needs to have reward 1.0

import os 
os.environ.setdefault("MUJOCO_GL", "egl")

from capx.envs.tasks import CodeExecEnvConfig, CodeExecutionEnvBase, get_exec_env, list_exec_envs, get_config, list_configs
from capx.envs.base import list_envs
from capx.integrations.base_api import list_apis
import tyro
import time

def test_environments(
    env_name: str = "franka_pick_place_code_env",
) -> bool:

    print("Available environments: ", list_envs())
    print("Available execution environments: ", list_exec_envs())
    print("Available APIs: ", list_apis())
    print("Available configurations: ", list_configs())
    cfg = get_config(env_name)
    # should be the same as 
    # cfg = CodeExecEnvConfig(
    #     low_level="franka_cubes_low_level",
    #     apis=["FrankaControlApi"],
    # )
    env : CodeExecutionEnvBase = get_exec_env(env_name)(cfg)
    env.enable_video_capture(True)
    start = time.time()
    obs, info = env.reset()
    end = time.time()
    print("Time taken to reset: ", end - start)
    print("Observation keys: ", list(obs.keys()))
    print("Prompt: ", obs["full_prompt"][1]["content"])
    start = time.time()
    obs_next, reward, terminated, truncated, info_step = env.step(env.oracle_code)
    end = time.time()
    print("Time taken: ", end - start)
    # package the video frames into a video
    video_frames = env.get_video_frames()
    if video_frames:
        import imageio
        imageio.mimsave("test_video.mp4", video_frames, fps=30)
        print("Video saved to test_video.mp4")
    if reward != 1.0:     
        # print("Observation: ", obs_next)
        print("Reward: ", reward)
        print("Terminated: ", terminated)
        print("Truncated: ", truncated)
        print("Info: ", info_step)
        return False
    else:
        print("Success")
        return True

def test_franka_pick_place_code_env() -> None:
    assert test_environments("franka_pick_place_code_env")

def test_franka_robosuite_pick_place_code_env() -> None:
    assert test_environments("franka_robosuite_pick_place_code_env")

def test_franka_lift_code_env() -> None:
    assert test_environments("franka_lift_code_env")
    
def test_franka_nut_assembly_code_env() -> None:
    assert test_environments("franka_nut_assembly_code_env")
    
def test_franka_pick_place_multi_code_env() -> None:
    assert test_environments("franka_pick_place_multi_code_env")
    
def test_r1pro_radio_code_env() -> None:
    assert test_environments("r1pro_radio_code_env")

def test_franka_libero_pick_place_code_env() -> None:
    assert test_environments("franka_libero_pick_place_code_env")

def test_franka_libero_pick_place_code_env_privileged() -> None:
    assert test_environments("franka_libero_pick_place_code_env_privileged")

def test_franka_libero_open_microwave_code_env() -> None:
    assert test_environments("franka_libero_open_microwave_code_env")

def test_franka_libero_open_microwave_code_env_privileged() -> None:
    assert test_environments("franka_libero_open_microwave_code_env_privileged")

if __name__ == "__main__":
    tyro.cli(test_environments)