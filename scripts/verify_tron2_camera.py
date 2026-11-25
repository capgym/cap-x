"""阶段 1.3（严谨版）：用 mujoco.mj_ray 求同一表面点真值，检验亚毫米级反投影误差。"""
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
XML_PATH = Path("capx/assets/tron2/DACH_TRON2A/xml/scene_table_cube.xml")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
H, W = 480, 640
R_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])
def get_camera_intrinsics(model: mujoco.MjModel, cam_id: int, height: int, width: int) -> np.ndarray:
    fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
def get_transforms(model: mujoco.MjModel, data: mujoco.MjData, cam_id: int):
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
    t_world_base = np.eye(4)
    t_world_base[:3, :3] = data.xmat[base_id].reshape(3, 3)
    t_world_base[:3, 3] = data.xpos[base_id]
    t_world_cam_cv = np.eye(4)
    t_world_cam_cv[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ R_MJ_TO_CV
    t_world_cam_cv[:3, 3] = data.cam_xpos[cam_id]
    t_base_cam_cv = np.linalg.inv(t_world_base) @ t_world_cam_cv
    return t_world_base, t_world_cam_cv, t_base_cam_cv
def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=H, width=W)
    for cam_name in ["robot0_robotview", "chest_d435"]:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        K = get_camera_intrinsics(model, cam_id, H, W)
        t_world_base, t_world_cam_cv, pose_mat = get_transforms(model, data, cam_id)
        t_base_world = np.linalg.inv(t_world_base)
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render()
        Image.fromarray(rgb).save(OUT_DIR / f"cam_{cam_name}.png")
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        depth = renderer.render()
        mask = (rgb[:, :, 0] > 160) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            print(f"[{cam_name}] 画面里没看到红方块！")
            continue
        # 选中方块掩码内的一个代表像素（OpenGL 像素中心在 u+0.5, v+0.5）
        u = int(np.median(xs))
        v = int(np.median(ys))
        d = float(depth[v, u])
        # 1) 反投影算出的表面点 p_hat_S (base_Link 坐标系)
        xn = (u + 0.5 - K[0, 2]) / K[0, 0]
        yn = (v + 0.5 - K[1, 2]) / K[1, 1]
        p_cam = np.array([xn * d, yn * d, d, 1.0])
        p_hat_base = (pose_mat @ p_cam)[:3]
        # 2) 用 mj_ray 沿同一像素视线求仿真几何真实命中的表面点 p_true_S (base_Link 坐标系)
        ray_cam = np.array([xn, yn, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = t_world_cam_cv[:3, :3] @ ray_cam
        cam_pos_world = t_world_cam_cv[:3, 3].copy()
        geomid = np.array([-1], dtype=np.int32)
        t_hit = mujoco.mj_ray(model, data, cam_pos_world, ray_world, None, 1, -1, geomid)
        hit_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0]))
        p_true_world = cam_pos_world + t_hit * ray_world
        p_true_base = (t_base_world @ np.append(p_true_world, 1.0))[:3]
        # 3) 同一个物理表面点 S 的真正反投影误差（毫米）
        err_mm = np.linalg.norm(p_hat_base - p_true_base) * 1000.0
        print(
            f"[{cam_name}] 像素=({u},{v}), 命中几何={hit_name}, 深度={d:.4f}m\n"
            f"    反投影表面点 p_hat_S  (base): {np.round(p_hat_base, 5)} m\n"
            f"    射线真值表面点 p_true_S (base): {np.round(p_true_base, 5)} m\n"
            f"    真实反投影误差 = {err_mm:.3f} mm"
        )
    renderer.close()
    print(f"已保存相机图片到: {OUT_DIR.resolve()}/cam_*.png")
if __name__ == "__main__":
    main()