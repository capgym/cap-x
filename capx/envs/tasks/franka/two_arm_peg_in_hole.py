from capx.envs.tasks.franka.franka_nut_assembly import FrankaNutAssemblyCodeEnv


PROMPT = """
You are controlling two opposed Franka Panda arms with the APIs described below. Arm 0 rigidly holds a cylindrical peg and arm 1 rigidly holds a plate with a hole.
Goal: move the peg into the hole with correct alignment.
Query the current peg pose, the arm-0 end-effector pose, and the documented hole insertion target pose before moving. Use relative_pose to express the arm-0 end effector relative to the peg. Compose that relative pose with the hole insertion target to obtain the desired arm-0 end-effector pose, then move arm 0 there. Arm 1 may remain fixed. Quaternions are WXYZ.
Output only one compact executable Python program without code fences. Call the provided APIs directly. Only numpy and scipy imports are allowed. Do not redefine APIs, access the environment directly, import robot-control modules, or use try/except.
"""


class TwoArmPegInHoleCodeEnv(FrankaNutAssemblyCodeEnv):
    prompt = PROMPT
    oracle_code = None


__all__ = ["TwoArmPegInHoleCodeEnv"]
