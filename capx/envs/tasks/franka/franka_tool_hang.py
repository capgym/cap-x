from capx.envs.tasks.franka.franka_nut_assembly import FrankaNutAssemblyCodeEnv


PROMPT = """
You are controlling a Franka Emika robot with the APIs described below.
Goal: first insert the hook frame into the stand mount, then hang the wrench on the assembled frame.
For the frame, query the frame tip pose and sample the frame grip pose before moving. Express the frame grip relative to the frame tip with relative_pose, grasp the frame grip from above, and compose that relative pose with the stand mount pose to obtain the desired frame-grip pose for insertion. Return home for a stable IK seed before insertion, approach from above, lower gently, and release.
Then query the wrench hole and sample the tool grip pose. Express the tool grip relative to the wrench hole, grasp the tool grip, compose that relative pose with the frame hang-site pose, approach the resulting tool-grip pose from above, lower gently so the hole goes over the hook, and release.
You may write Python comments for reasoning, but output only one compact executable Python program and do not use code fences.
Call the provided APIs directly. Only numpy and scipy imports are allowed. Do not redefine APIs, import robot-control modules, access the environment directly, or use try/except.
"""


class FrankaToolHangCodeEnv(FrankaNutAssemblyCodeEnv):
    prompt = PROMPT
    oracle_code = None


__all__ = ["FrankaToolHangCodeEnv"]
