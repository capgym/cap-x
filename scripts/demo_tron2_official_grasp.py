"""阶段 2.2 实时联调演示：Mac 本地桌面 MuJoCo 场景 + 4090w 官方 Tron2_controller (MoveJ / MoveP)。

运行前确认：
1. 4090w 已启动仿真器 + 官方控制器，并切到 develop 模式；
2. Mac 已建立 5000 端口隧道：ssh -N -L 5000:127.0.0.1:5000 4090w

运行方式：
    uv run --no-sync python scripts/demo_tron2_official_grasp.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# 直接从 capx/integrations/tron2 目录导入，避免触发顶层 capx/__init__.py 拉起 cv2/viser/robosuite 等重依赖
REPO_ROOT = Path(__file__).resolve().parents[1]
TRON2_DIR = REPO_ROOT / "capx/integrations/tron2"
if str(TRON2_DIR) not in sys.path:
    sys.path.insert(0, str(TRON2_DIR))

from official_client import (
    DEFAULT_HOME_Q16,
    HORIZONTAL_GRASP_QUAT_WXYZ,
    PROTOCOL_JOINT_NAMES,
    Tron2OfficialClient,
)

XML_PATH = REPO_ROOT / "capx/assets/tron2/DACH_TRON2A/xml/scene_table_cube.xml"

# 16 个官方受控轴的局部追踪 PD（仅用于把官方算出的 q_ref 同步到桌面场景）
ARM_GAINS = [
    (400.0, 30.0), (400.0, 30.0), (250.0, 18.0), (300.0, 22.0),
    (120.0, 8.0), (120.0, 8.0), (120.0, 8.0),
    (400.0, 30.0), (400.0, 30.0), (250.0, 18.0), (300.0, 22.0),
    (120.0, 8.0), (120.0, 8.0), (120.0, 8.0),
    (80.0, 6.0), (80.0, 6.0),
]

# 4 个夹爪电机：底座稳在 0 位，drive 轴控制手指张开 (0.72 rad) / 闭合夹紧 (0.0 rad)
GRIPPER_OPEN_RAD = 0.72
GRIPPER_CLOSE_RAD = 0.0


def build_joint_map(model: mujoco.MjModel) -> dict[str, tuple[int, int, int]]:
    mapping: dict[str, tuple[int, int, int]] = {}
    for act_id in range(model.nu):
        jnt_id = int(model.actuator_trnid[act_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if name:
            mapping[name] = (int(model.jnt_qposadr[jnt_id]), int(model.jnt_dofadr[jnt_id]), act_id)
    return mapping


def main() -> None:
    print("1. 连接 4090w 官方 Tron2_controller (ws://127.0.0.1:5000)...")
    client = Tron2OfficialClient(host="127.0.0.1", port=5000, accid="DACH_TRON2A_001")
    client.connect()
    q_init = client.get_joint_state()
    print(f"   已连接！当前左臂 7 轴: {np.round(q_init[:7], 3)}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)

    jnt_map = build_joint_map(model)
    arm_bindings = []
    for idx, name in enumerate(PROTOCOL_JOINT_NAMES):
        qpos_adr, dof_adr, act_id = jnt_map[name]
        kp, kd = ARM_GAINS[idx]
        lo, hi = model.actuator_ctrlrange[act_id]
        arm_bindings.append((idx, qpos_adr, dof_adr, act_id, kp, kd, lo, hi))

    # 分别绑定左右夹爪底座与驱动电机
    grip_base_l = (*jnt_map["grasper_base_L_Joint"], 12.0, 0.5, *model.actuator_ctrlrange[jnt_map["grasper_base_L_Joint"][2]])
    grip_base_r = (*jnt_map["grasper_base_R_Joint"], 12.0, 0.5, *model.actuator_ctrlrange[jnt_map["grasper_base_R_Joint"][2]])
    grip_drv_l = (*jnt_map["grasper_L_drive_Joint"], 4.0, 0.15, *model.actuator_ctrlrange[jnt_map["grasper_L_drive_Joint"][2]])
    grip_drv_r = (*jnt_map["grasper_R_drive_Joint"], 4.0, 0.15, *model.actuator_ctrlrange[jnt_map["grasper_R_drive_Joint"][2]])

    # 用官方控制器当前角度初始化本地桌面场景，避免 t=0 突变
    for idx, qpos_adr, *_ in arm_bindings:
        data.qpos[qpos_adr] = q_init[idx]
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    cube_init_world = data.xpos[cube_id].copy()
    cube_pos_base = cube_init_world - data.xpos[base_id]
    print(f"2. 红方块初始位置 (base_Link): {np.round(cube_pos_base, 4)} m")

    # 线程共享目标（由后台控制线程持续从官方控制器拉取 q_ref）
    lock = threading.Lock()
    shared = {
        "q_ref": q_init.copy(),
        "grip_l": GRIPPER_OPEN_RAD,
        "grip_r": 0.0,
        "stop": False,
    }

    def poll_for(seconds: float) -> None:
        end_t = time.perf_counter() + seconds
        while not shared["stop"] and time.perf_counter() < end_t:
            q_now = client.get_joint_state()
            with lock:
                shared["q_ref"] = q_now
            time.sleep(0.025)  # ~40 Hz 实时同步

    def run_grasp_sequence() -> None:
        try:
            # 等待窗口打开 1 秒，方便你眼睛聚焦到红方块
            poll_for(1.0)

            right_park = [0.46, -0.32, -0.20]
            quat_h = HORIZONTAL_GRASP_QUAT_WXYZ

            print("\n[步骤 1/5] MoveJ -> J_PREP（双臂抬起到准备姿势，同时张开左手夹爪）")
            with lock:
                shared["grip_l"] = GRIPPER_OPEN_RAD
            client.move_j(DEFAULT_HOME_Q16[:14], duration_sec=2.5)
            poll_for(3.0)

            # 读取红方块在 base_Link 下的坐标，确定预接近 / 抓取 / 抬起三段 MoveP 目标
            grasp_z = float(cube_pos_base[2]) + 0.010  # 略高于方块中心 1cm，稳妥避开桌面
            pre_grasp = [float(cube_pos_base[0]) - 0.11, float(cube_pos_base[1]), grasp_z]
            at_grasp = [float(cube_pos_base[0]), float(cube_pos_base[1]), grasp_z]
            lift_pos = [float(cube_pos_base[0]), float(cube_pos_base[1]), grasp_z + 0.16]

            print(f"[步骤 2/5] MoveP -> 预接近点 {np.round(pre_grasp, 3)}（方块后方 11 cm）")
            resp1 = client.move_p(pre_grasp, quat_h, right_park, quat_h, duration_sec=2.0)
            print(f"   官方规划反馈: {resp1.get('message', resp1)}")
            poll_for(2.5)

            print(f"[步骤 3/5] MoveP -> 水平前伸到方块位置 {np.round(at_grasp, 3)}")
            resp2 = client.move_p(at_grasp, quat_h, right_park, quat_h, duration_sec=1.8)
            print(f"   官方规划反馈: {resp2.get('message', resp2)}")
            poll_for(2.3)

            print("[步骤 4/5] 闭合左手夹爪（夹紧 5 cm 红方块）")
            for alpha in np.linspace(0.0, 1.0, 25):
                with lock:
                    shared["grip_l"] = (1.0 - alpha) * GRIPPER_OPEN_RAD + alpha * GRIPPER_CLOSE_RAD
                poll_for(0.04)
            poll_for(0.6)

            print(f"[步骤 5/5] MoveP -> 竖直抬高 16 cm 到 {np.round(lift_pos, 3)}")
            resp3 = client.move_p(lift_pos, quat_h, right_park, quat_h, duration_sec=2.0)
            print(f"   官方规划反馈: {resp3.get('message', resp3)}")
            poll_for(2.6)

            cube_now = data.xpos[cube_id].copy()
            dz_cm = (cube_now[2] - cube_init_world[2]) * 100.0
            print(f"\n=== 抓取结果：红方块竖直抬升了 {dz_cm:.2f} cm！（{'成功！' if dz_cm > 8.0 else '未抬起，需微调抓取点'}）===")
            print("你可以继续旋转/缩放 3D 窗口仔细检查夹持状态，关闭窗口即可退出。")

            while not shared["stop"]:
                poll_for(0.2)
        except Exception as exc:
            print(f"\n[错误] 抓取流程异常: {exc}", flush=True)

    def ctrl_callback(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        with lock:
            q_ref = shared["q_ref"]
            grip_l = shared["grip_l"]
            grip_r = shared["grip_r"]

        for idx, qpos_adr, dof_adr, act_id, kp, kd, lo, hi in arm_bindings:
            tau = kp * (q_ref[idx] - d.qpos[qpos_adr]) + kd * (0.0 - d.qvel[dof_adr]) + d.qfrc_bias[dof_adr]
            d.ctrl[act_id] = float(np.clip(tau, lo, hi))

        for (qpos_adr, dof_adr, act_id, kp, kd, lo, hi), target in [
            (grip_base_l, 0.0),
            (grip_base_r, 0.0),
            (grip_drv_l, grip_l),
            (grip_drv_r, grip_r),
        ]:
            tau = kp * (target - d.qpos[qpos_adr]) + kd * (0.0 - d.qvel[dof_adr]) + d.qfrc_bias[dof_adr]
            d.ctrl[act_id] = float(np.clip(tau, lo, hi))

    worker = threading.Thread(target=run_grasp_sequence, daemon=True)
    worker.start()

    mujoco.set_mjcb_control(ctrl_callback)
    try:
        mujoco.viewer.launch(model, data)
    finally:
        shared["stop"] = True
        worker.join(timeout=1.0)
        client.close()


if __name__ == "__main__":
    main()
