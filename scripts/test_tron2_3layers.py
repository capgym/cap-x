"""第 4.1 步：在本地 Mac 端到端验收 TRON2 三层 Cap-X 架构（Env + API + ORACLE_CODE + 录像）。"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# 仅用于本地 Mac 轻环境：跳过原版 franka 包的 cv2/torch 顶层导入，直接加载你写的 TRON2 三层
for pkg, rel in [
    ("capx", "capx"),
    ("capx.envs", "capx/envs"),
    ("capx.envs.simulators", "capx/envs/simulators"),
    ("capx.integrations", "capx/integrations"),
    ("capx.integrations.tron2", "capx/integrations/tron2"),
]:
    mod = types.ModuleType(pkg)
    mod.__path__ = [str(ROOT / rel)]
    sys.modules[pkg] = mod

from capx.envs.simulators.tron2_cube_lift import Tron2CubeLiftLowLevel
from capx.integrations.tron2.control_privileged import Tron2ControlPrivilegedApi

# 直接从你第 3 步写的任务文件里提取 ORACLE_CODE 字符串
task_src = (ROOT / "capx/envs/tasks/tron2/tron2_cube_lift.py").read_text(encoding="utf-8")
tree = ast.parse(task_src)
oracle_code = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "ORACLE_CODE":
                oracle_code = ast.literal_eval(node.value)
assert isinstance(oracle_code, str), "未在 tron2_cube_lift.py 找到 ORACLE_CODE"


def main() -> None:
    env = Tron2CubeLiftLowLevel(privileged=True, enable_render=True)
    api = Tron2ControlPrivilegedApi(env)
    try:
        print("=== 给大模型看的操作说明书 (api.combined_doc) 预览 ===")
        print(api.combined_doc()[:400], "...\n")

        print("1. 调用 env.reset()（官方 MoveJ 归位 + 放好红方块）...")
        env.reset()
        env.enable_video_capture(True)

        print("2. 按 Cap-X 方式执行 ORACLE_CODE（把 api.functions() 注入执行环境）...")
        exec_globals = {"__builtins__": __builtins__, **api.functions()}
        exec(oracle_code, exec_globals, exec_globals)

        reward = env.compute_reward()
        completed = env.task_completed()
        frames = env.get_video_frames()
        print(f"\n=== 验收结果 ===")
        print(f"  reward         = {reward:.3f}")
        print(f"  task_completed = {completed}")
        print(f"  录制视频帧数    = {len(frames)}")

        if frames:
            out_gif = ROOT / "outputs/tron2_oracle_lift.gif"
            out_gif.parent.mkdir(exist_ok=True)
            pil_frames = [Image.fromarray(f).resize((480, 360)) for f in frames[::2]]
            pil_frames[0].save(
                out_gif,
                save_all=True,
                append_images=pil_frames[1:],
                duration=100,
                loop=0,
            )
            print(f"  已导出录像动图: {out_gif}")
    finally:
        env.close()


if __name__ == "__main__":
    main()