from capx.envs.tasks.franka.franka_nut_assembly import FrankaNutAssemblyCodeEnv


PROMPT = """
You are controlling a Franka Emika robot with the APIs described below.
Goal: first insert the hook frame into the stand mount, then hang the wrench on the assembled frame.
For the frame, query the current frame tip pose, the documented frame insertion target, and the current frame grip pose before moving. Express the frame grip relative to the frame tip with relative_pose. You must first move to the current frame grip with z_approach, close the gripper, and return home while holding it. Then compose the frame insertion target with the saved relative pose, move to that resulting frame-grip pose with z_approach, and release. Never close the gripper at the insertion target without first grasping the frame at its current location.
Then query the current wrench hole, the documented tool hanging target, and the current tool grip. Express the tool grip relative to the wrench hole. You must move to the current tool grip with z_approach, close the gripper, and return home while holding it. Compose the tool hanging target with the saved relative pose, move to the resulting tool-grip pose with z_approach, and release. Never close the gripper at the hanging target without first grasping the wrench at its current location.
You may write Python comments for reasoning, but output only one compact executable Python program and do not use code fences.
Call the provided APIs directly. Only numpy and scipy imports are allowed. Do not redefine APIs, import robot-control modules, access the environment directly, or use try/except.
"""


class FrankaToolHangCodeEnv(FrankaNutAssemblyCodeEnv):
    prompt = PROMPT
    oracle_code = None


__all__ = ["FrankaToolHangCodeEnv"]
