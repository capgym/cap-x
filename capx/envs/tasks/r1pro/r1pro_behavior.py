from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a R1Pro robot with API described below.
Goal: Complete the task described in the environment.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
If you want to use numpy, scipy for spatial transformations, opencv, pytorch, or any other libraries, you need to import them explicitly.
Note that API may fail. Make sure the code is fault tolerant.
You should consider retrying, try and except, and retrying other combinations of APIs or write your own code to recreate the same capability.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""


# ---------------------------- High-level Env -----------------------------
class R1ProBehaviorCodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for R1Pro BEHAVIOR tasks using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = None


__all__ = [
    "R1ProBehaviorCodeEnv",
]
