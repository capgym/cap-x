from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation


DEFAULT_CONFIG_PATH = "env_configs/g1/g1_grasp_bottle.yaml"
DEFAULT_OBJECT_NAME = "plastic water bottle"
DEFAULT_OUTPUT_DIR = "./outputs/g1_grasp_bottle_preflight"
DEFAULT_SAM3_URL = "http://127.0.0.1:8114"
DEFAULT_GRASPNET_URL = "http://127.0.0.1:8115"
DEFAULT_PYROKI_URL = "http://127.0.0.1:8116"
RIGHT_ARM_STICK_LINKS = (
    "torso_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_hand_palm_link",
)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_builtin(data), indent=2), encoding="utf-8")


def _normalize_depth(depth: Any) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
        depth_arr = depth_arr[:, :, 0]
    if depth_arr.ndim != 2:
        raise ValueError(f"Expected depth shape HxW or HxWx1, got {depth_arr.shape}.")
    return np.ascontiguousarray(depth_arr)


def _normalize_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != shape:
        raise ValueError(f"SAM3 mask shape {mask_arr.shape} does not match depth shape {shape}.")
    return mask_arr


def deproject_masked_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Deproject masked RGB-D pixels to camera-frame points and RGB colors."""

    depth_arr = _normalize_depth(depth)
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    if rgb_arr.shape[:2] != depth_arr.shape:
        raise ValueError(f"RGB shape {rgb_arr.shape[:2]} does not match depth shape {depth_arr.shape}.")
    mask_arr = _normalize_mask(mask, depth_arr.shape)
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"Intrinsics must be 3x3, got {k.shape}.")

    valid = (
        mask_arr
        & np.isfinite(depth_arr)
        & (depth_arr >= float(depth_clip_range[0]))
        & (depth_arr <= float(depth_clip_range[1]))
    )
    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[:: int(stride), :: int(stride)] = True
        valid &= stride_mask

    ys, xs = np.where(valid)
    z = depth_arr[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    points = np.stack([x, y, z], axis=1)
    colors = rgb_arr[ys, xs, :3].astype(np.float64) / 255.0
    return points, colors


def transform_pose_matrix(
    t_world_camera: np.ndarray,
    t_camera_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Transform a camera-frame pose matrix to world-frame matrix and wxyz/xyz pose."""

    world_camera = np.asarray(t_world_camera, dtype=np.float64).reshape(4, 4)
    camera_target = np.asarray(t_camera_target, dtype=np.float64).reshape(4, 4)
    world_target = world_camera @ camera_target
    quat_xyzw = SciRotation.from_matrix(world_target[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return world_target, {
        "position": world_target[:3, 3].astype(float).tolist(),
        "quaternion_wxyz": quat_wxyz.astype(float).tolist(),
    }


def extract_right_arm_ik_solution(cfg: Any) -> np.ndarray:
    arr = np.asarray(cfg, dtype=np.float64).reshape(-1)
    if arr.size == 7:
        return arr
    if arr.size == 43:
        return arr[29:36]
    if arr.size >= 14:
        return arr[7:14]
    raise ValueError(f"Expected a 7, 14+, or 43 DoF IK solution, got shape {np.asarray(cfg).shape}.")


def compute_right_arm_stick_figure(
    *,
    robot_urdf_path: str,
    root_link_name: str,
    target_link_name: str,
    joints: Any,
) -> dict[str, Any]:
    from capx.integrations.motion.pyroki_context import get_pyroki_context

    ctx = get_pyroki_context(
        robot_urdf_path,
        target_link_name=target_link_name,
        root_link_name=root_link_name,
    )
    q = extract_right_arm_ik_solution(joints)
    if q.size != ctx.robot.joints.num_actuated_joints:
        raise ValueError(
            f"IK solution has {q.size} joints but PyRoKi right-arm model expects "
            f"{ctx.robot.joints.num_actuated_joints}."
        )

    fk = np.asarray(ctx.robot.forward_kinematics(q), dtype=np.float64)
    link_positions: list[dict[str, Any]] = []
    link_names = tuple(ctx.robot.links.names)
    for link_name in RIGHT_ARM_STICK_LINKS:
        if link_name not in link_names:
            continue
        pose_wxyz_xyz = fk[link_names.index(link_name)]
        link_positions.append(
            {
                "name": link_name,
                "position": pose_wxyz_xyz[4:7].astype(float).tolist(),
            }
        )

    present_names = {item["name"] for item in link_positions}
    segments: list[list[str]] = []
    previous_name: str | None = None
    for link_name in RIGHT_ARM_STICK_LINKS:
        if link_name not in present_names:
            continue
        if previous_name is not None:
            segments.append([previous_name, link_name])
        previous_name = link_name

    return {
        "frame": root_link_name,
        "root_link": root_link_name,
        "target_link": target_link_name,
        "joint_names": list(ctx.robot.joints.actuated_names),
        "joint_positions": q.astype(float).tolist(),
        "links": link_positions,
        "segments": segments,
    }


def _pose_matrix_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def _translation_matrix(offset_xyz: list[float] | tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(offset_xyz, dtype=np.float64).reshape(3)
    return pose


def _save_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import open3d as o3d

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pointcloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pointcloud)


def _compute_obb_pose(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import open3d as o3d

    if len(points) < 8:
        raise ValueError(f"Need at least 8 segmented 3D points for OBB pose, got {len(points)}.")
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if len(points) >= 25:
        pointcloud, _ = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    obb = pointcloud.get_oriented_bounding_box()
    pose = _pose_matrix_from_rotation_translation(obb.R, obb.center)
    return pose, {
        "center": np.asarray(obb.center, dtype=float).tolist(),
        "rotation_matrix": np.asarray(obb.R, dtype=float).tolist(),
        "extent": np.asarray(obb.extent, dtype=float).tolist(),
        "num_points": int(len(pointcloud.points)),
    }


def _best_sam3_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("SAM3 returned no detections.")
    return max(results, key=lambda item: float(item.get("score", 0.0)))


def _summarize_sam3(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        mask = np.asarray(result.get("mask", []), dtype=bool)
        summary.append(
            {
                "index": idx,
                "label": str(result.get("label", "")),
                "score": float(result.get("score", 0.0)),
                "box": [float(v) for v in result.get("box", [])],
                "mask_area_px": int(mask.sum()) if mask.size else 0,
            }
        )
    return summary


def _save_sam3_overlay(rgb: np.ndarray, best: dict[str, Any], path: Path) -> None:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask = np.asarray(best["mask"], dtype=bool)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 120), mode="L")
    color = Image.new("RGBA", image.size, (20, 170, 255, 0))
    color.putalpha(mask_img)
    overlay.alpha_composite(color)
    draw = ImageDraw.Draw(overlay)
    box = [float(v) for v in best.get("box", [])]
    if len(box) == 4:
        draw.rectangle(box, outline=(255, 210, 0, 255), width=3)
    image.alpha_composite(overlay)
    image.convert("RGB").save(path)


def _image_to_base64(path: str | Path | None) -> str | None:
    if path is None:
        return None
    img_path = Path(path)
    if not img_path.exists():
        return None
    suffix = img_path.suffix.lower().lstrip(".") or "png"
    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{data}"


def _point_preview(points: np.ndarray, colors: np.ndarray, *, max_points: int = 12000) -> dict[str, Any]:
    if len(points) == 0:
        return {"points": [], "colors": []}
    step = max(1, int(np.ceil(len(points) / max_points)))
    return {
        "points": np.asarray(points[::step], dtype=np.float32).round(5).tolist(),
        "colors": np.asarray(colors[::step], dtype=np.float32).round(4).tolist(),
    }


def _scene_preview(summary: dict[str, Any], point_preview: dict[str, Any]) -> dict[str, Any]:
    return {
        "pointCloud": point_preview,
        "graspPoseMatrix": summary.get("grasp_pose_world_matrix"),
        "approachPosition": summary.get("ik", {}).get("approach_position_world"),
        "objectPoseMatrix": summary.get("object_modal_pose_world_matrix"),
        "cameraPoseMatrix": summary.get("camera", {}).get("T_world_camera"),
        "ikStickFigure": summary.get("ik", {}).get("final_stick_figure"),
    }


def build_report_html(
    summary: dict[str, Any],
    artifacts: dict[str, str],
    *,
    point_preview: dict[str, Any],
) -> str:
    summary_json = json.dumps(_to_builtin(summary), indent=2)
    artifacts_json = json.dumps(_to_builtin(artifacts), indent=2)
    scene_json = json.dumps(_to_builtin(_scene_preview(summary, point_preview)), separators=(",", ":"))
    sam3_overlay_uri = _image_to_base64(artifacts.get("sam3_overlay"))
    depth_uri = _image_to_base64(artifacts.get("depth_color"))
    rgb_uri = _image_to_base64(artifacts.get("rgb"))
    object_name = html.escape(str(summary.get("object_name", "")))

    def _img(src: str | None, label: str) -> str:
        if not src:
            return ""
        return f'<figure><img src="{src}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>G1 Grasp Bottle Preflight</title>
  <style>
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; color: #202124; background: #f7f7f4; }}
    header {{ padding: 20px 28px; background: #ffffff; border-bottom: 1px solid #d8d8d0; }}
    h1 {{ margin: 0; font-size: 22px; font-weight: 650; letter-spacing: 0; }}
    main {{ padding: 22px 28px 36px; display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; }}
    section {{ background: #ffffff; border: 1px solid #d8d8d0; border-radius: 6px; padding: 16px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; font-weight: 650; letter-spacing: 0; }}
    .images {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; border: 1px solid #d8d8d0; border-radius: 4px; background: #111; }}
    figcaption {{ margin-top: 5px; color: #5f6368; font-size: 12px; }}
    pre {{ overflow: auto; max-height: 420px; padding: 12px; background: #f2f2ee; border-radius: 4px; font-size: 12px; line-height: 1.42; }}
    canvas {{ width: 100%; height: 460px; border: 1px solid #d8d8d0; border-radius: 4px; background: #111; display: block; }}
    .controls {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; font-size: 13px; }}
    input[type="range"] {{ width: 160px; }}
    .wide {{ grid-column: 1 / -1; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns: 1fr; padding: 16px; }} .images {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><h1>G1 Grasp Bottle Preflight: {object_name}</h1></header>
  <main>
    <section class="wide">
      <h2>Camera And SAM3</h2>
      <div class="images">
        {_img(rgb_uri, "RGB")}
        {_img(depth_uri, "Depth")}
        {_img(sam3_overlay_uri, "SAM3 Overlay")}
      </div>
    </section>
    <section>
      <h2>3D Grasp Preview</h2>
      <div class="controls">
        <label>Yaw <input id="yaw" type="range" min="-180" max="180" value="35"></label>
        <label>Pitch <input id="pitch" type="range" min="-90" max="90" value="-20"></label>
        <label>Roll <input id="roll" type="range" min="-180" max="180" value="0"></label>
        <label>Zoom <input id="zoom" type="range" min="60" max="900" value="360"></label>
      </div>
      <canvas id="pc"></canvas>
    </section>
    <section>
      <h2>Summary</h2>
      <pre>{html.escape(summary_json)}</pre>
    </section>
    <section class="wide">
      <h2>Artifacts</h2>
      <pre>{html.escape(artifacts_json)}</pre>
    </section>
  </main>
  <script>
    const scene = {scene_json};
    const preview = scene.pointCloud || {{}};
    const graspPoseMatrix = scene.graspPoseMatrix;
    const approachPosition = scene.approachPosition;
    const objectPoseMatrix = scene.objectPoseMatrix;
    const cameraPoseMatrix = scene.cameraPoseMatrix;
    const ikStickFigure = scene.ikStickFigure || {{}};
    const canvas = document.getElementById('pc');
    const ctx = canvas.getContext('2d');
    const yawEl = document.getElementById('yaw');
    const pitchEl = document.getElementById('pitch');
    const rollEl = document.getElementById('roll');
    const zoomEl = document.getElementById('zoom');
    function resize() {{
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(r.width * devicePixelRatio));
      canvas.height = Math.max(320, Math.floor(r.height * devicePixelRatio));
    }}
    function isVec3(v) {{
      return Array.isArray(v) && v.length >= 3 && v.every(Number.isFinite);
    }}
    function matrixTranslation(m) {{
      if (!Array.isArray(m) || m.length < 3) return null;
      const p = [m[0]?.[3], m[1]?.[3], m[2]?.[3]];
      return isVec3(p) ? p : null;
    }}
    function transformLocal(m, local) {{
      if (!Array.isArray(m) || m.length < 3) return null;
      const p = [
        m[0][0] * local[0] + m[0][1] * local[1] + m[0][2] * local[2] + m[0][3],
        m[1][0] * local[0] + m[1][1] * local[1] + m[1][2] * local[2] + m[1][3],
        m[2][0] * local[0] + m[2][1] * local[1] + m[2][2] * local[2] + m[2][3],
      ];
      return isVec3(p) ? p : null;
    }}
    function identityMatrix() {{
      return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    }}
    function stickFigurePoints(stick) {{
      const links = Array.isArray(stick?.links) ? stick.links : [];
      return links.map((link) => link.position).filter(isVec3);
    }}
    function sceneCenter(pts) {{
      const anchors = pts.slice();
      const graspOrigin = matrixTranslation(graspPoseMatrix);
      const objectOrigin = matrixTranslation(objectPoseMatrix);
      const cameraOrigin = matrixTranslation(cameraPoseMatrix);
      if (graspOrigin) anchors.push(graspOrigin);
      if (objectOrigin) anchors.push(objectOrigin);
      if (cameraOrigin) anchors.push(cameraOrigin);
      if (isVec3(approachPosition)) anchors.push(approachPosition);
      anchors.push([0, 0, 0]);
      for (const p of stickFigurePoints(ikStickFigure)) anchors.push(p);
      if (!anchors.length) return [0, 0, 0];
      let cx = 0, cy = 0, cz = 0;
      for (const p of anchors) {{ cx += p[0]; cy += p[1]; cz += p[2]; }}
      return [cx / anchors.length, cy / anchors.length, cz / anchors.length];
    }}
    function makeView(center) {{
      const yaw = Number(yawEl.value) * Math.PI / 180;
      const pitch = Number(pitchEl.value) * Math.PI / 180;
      const roll = Number(rollEl.value) * Math.PI / 180;
      return {{
        center,
        zoom: Number(zoomEl.value) * devicePixelRatio,
        cy: Math.cos(yaw),
        sy: Math.sin(yaw),
        cp: Math.cos(pitch),
        sp: Math.sin(pitch),
        cr: Math.cos(roll),
        sr: Math.sin(roll),
      }};
    }}
    function projectPoint(p, view) {{
      const x0 = p[0] - view.center[0], y0 = p[1] - view.center[1], z0 = p[2] - view.center[2];
      const x1 = view.cy * x0 + view.sy * z0;
      const z1 = -view.sy * x0 + view.cy * z0;
      const y1 = view.cp * y0 - view.sp * z1;
      const z2 = view.sp * y0 + view.cp * z1 + 3.0;
      const x2 = view.cr * x1 - view.sr * y1;
      const y2 = view.sr * x1 + view.cr * y1;
      return {{
        x: canvas.width * 0.5 + x2 * view.zoom / Math.max(0.2, z2),
        y: canvas.height * 0.5 - y2 * view.zoom / Math.max(0.2, z2),
        z: z2,
      }};
    }}
    function drawLine3d(a, b, view, color, width = 2) {{
      if (!isVec3(a) || !isVec3(b)) return;
      const pa = projectPoint(a, view);
      const pb = projectPoint(b, view);
      ctx.strokeStyle = color;
      ctx.lineWidth = width * devicePixelRatio;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }}
    function drawMarker3d(p, view, color, radius = 6) {{
      if (!isVec3(p)) return;
      const pp = projectPoint(p, view);
      ctx.fillStyle = color;
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 1.5 * devicePixelRatio;
      ctx.beginPath();
      ctx.arc(pp.x, pp.y, radius * devicePixelRatio, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }}
    function drawLabel(text, p, view, color = '#f8f8f8') {{
      if (!isVec3(p)) return;
      const pp = projectPoint(p, view);
      ctx.fillStyle = color;
      ctx.font = `${{12 * devicePixelRatio}}px Inter, Arial, sans-serif`;
      ctx.fillText(text, pp.x + 9 * devicePixelRatio, pp.y - 9 * devicePixelRatio);
    }}
    function drawCoordinateFrame(m, view, label, axisLen = 0.08) {{
      const origin = matrixTranslation(m);
      if (!origin) return;
      drawLine3d(origin, transformLocal(m, [axisLen, 0, 0]), view, '#ff5a52', 2);
      drawLine3d(origin, transformLocal(m, [0, axisLen, 0]), view, '#46d36d', 2);
      drawLine3d(origin, transformLocal(m, [0, 0, axisLen]), view, '#55a7ff', 2);
      drawMarker3d(origin, view, '#f8f8f8', 4);
      drawLabel(label, origin, view, '#f8f8f8');
    }}
    function linkPositionMap(stick) {{
      const out = new Map();
      const links = Array.isArray(stick?.links) ? stick.links : [];
      for (const link of links) {{
        if (typeof link.name === 'string' && isVec3(link.position)) out.set(link.name, link.position);
      }}
      return out;
    }}
    function drawRobotStickFigure(view) {{
      const positions = linkPositionMap(ikStickFigure);
      if (!positions.size) return;
      const segments = Array.isArray(ikStickFigure.segments) ? ikStickFigure.segments : [];
      for (const seg of segments) {{
        if (!Array.isArray(seg) || seg.length < 2) continue;
        drawLine3d(positions.get(seg[0]), positions.get(seg[1]), view, '#31d0aa', 3);
      }}
      for (const [name, p] of positions.entries()) {{
        const isPalm = name === 'right_hand_palm_link';
        const isElbow = name === 'right_elbow_link';
        const isRoot = name === 'torso_link';
        drawMarker3d(p, view, isPalm ? '#ffffff' : isElbow ? '#ff9f43' : isRoot ? '#8ab4f8' : '#31d0aa', isPalm ? 6 : 4);
      }}
      const palm = positions.get('right_hand_palm_link');
      const shoulder = positions.get('right_shoulder_pitch_link');
      if (shoulder) drawLabel('IK right arm', shoulder, view, '#b7fff0');
      if (palm) drawLabel('IK palm', palm, view, '#ffffff');
    }}
    function drawGraspPose(view) {{
      const origin = matrixTranslation(graspPoseMatrix);
      if (!origin) return;
      const axisLen = 0.09;
      const xAxis = transformLocal(graspPoseMatrix, [axisLen, 0, 0]);
      const yAxis = transformLocal(graspPoseMatrix, [0, axisLen, 0]);
      const zAxis = transformLocal(graspPoseMatrix, [0, 0, axisLen]);
      if (isVec3(approachPosition)) {{
        drawLine3d(approachPosition, origin, view, '#ffcf33', 2.5);
        drawMarker3d(approachPosition, view, '#ffcf33', 4);
      }}
      drawLine3d(origin, xAxis, view, '#ff5a52', 3);
      drawLine3d(origin, yAxis, view, '#46d36d', 3);
      drawLine3d(origin, zAxis, view, '#55a7ff', 3);
      drawMarker3d(origin, view, '#ffffff', 6);
      drawLabel('Final hand pose', origin, view);
    }}
    function drawObjectPose(view) {{
      const origin = matrixTranslation(objectPoseMatrix);
      if (!origin) return;
      drawMarker3d(origin, view, '#b58cff', 4);
      drawLabel('object pose', origin, view, '#d6c8ff');
    }}
    function drawLegend() {{
      ctx.font = `${{12 * devicePixelRatio}}px Inter, Arial, sans-serif`;
      ctx.fillStyle = '#f8f8f8';
      ctx.fillText('Final hand pose', 18 * devicePixelRatio, 24 * devicePixelRatio);
      ctx.fillStyle = '#ffcf33';
      ctx.fillText('approach path', 18 * devicePixelRatio, 43 * devicePixelRatio);
      ctx.fillStyle = '#ff5a52';
      ctx.fillText('x', 18 * devicePixelRatio, 62 * devicePixelRatio);
      ctx.fillStyle = '#46d36d';
      ctx.fillText('y', 36 * devicePixelRatio, 62 * devicePixelRatio);
      ctx.fillStyle = '#55a7ff';
      ctx.fillText('z', 54 * devicePixelRatio, 62 * devicePixelRatio);
      ctx.fillStyle = '#31d0aa';
      ctx.fillText('IK right arm', 18 * devicePixelRatio, 81 * devicePixelRatio);
      ctx.fillStyle = '#8ab4f8';
      ctx.fillText('torso/camera frames', 18 * devicePixelRatio, 100 * devicePixelRatio);
    }}
    function draw() {{
      resize();
      const pts = preview.points || [];
      const cols = preview.colors || [];
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const hasGrasp = Boolean(matrixTranslation(graspPoseMatrix));
      const hasRobot = stickFigurePoints(ikStickFigure).length > 0 || Boolean(matrixTranslation(cameraPoseMatrix));
      if (!pts.length && !hasGrasp && !hasRobot) {{
        ctx.fillStyle = '#ddd';
        ctx.fillText('No point cloud preview points', 24, 32);
        return;
      }}
      const view = makeView(sceneCenter(pts));
      for (let i = 0; i < pts.length; i++) {{
        const p = pts[i];
        const pp = projectPoint(p, view);
        const c = cols[i] || [0.2, 0.8, 1.0];
        ctx.fillStyle = `rgb(${{Math.floor(c[0]*255)}},${{Math.floor(c[1]*255)}},${{Math.floor(c[2]*255)}})`;
        ctx.fillRect(pp.x, pp.y, 2 * devicePixelRatio, 2 * devicePixelRatio);
      }}
      drawCoordinateFrame(identityMatrix(), view, 'torso_link', 0.12);
      drawCoordinateFrame(cameraPoseMatrix, view, 'camera', 0.08);
      drawRobotStickFigure(view);
      drawObjectPose(view);
      drawGraspPose(view);
      drawLegend();
    }}
    yawEl.addEventListener('input', draw);
    pitchEl.addEventListener('input', draw);
    rollEl.addEventListener('input', draw);
    zoomEl.addEventListener('input', draw);
    window.addEventListener('resize', draw);
    draw();
  </script>
</body>
</html>
"""


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "rgb": output_dir / "rgb.png",
        "depth": output_dir / "depth_m.npy",
        "depth_color": output_dir / "depth_color.png",
        "intrinsics": output_dir / "intrinsics.npy",
        "camera_pose": output_dir / "T_world_camera.npy",
        "sam3_overlay": output_dir / "sam3_overlay.png",
        "sam3_summary": output_dir / "sam3_summary.json",
        "full_pointcloud_npz": output_dir / "pointcloud_full_camera.npz",
        "segment_pointcloud_npz": output_dir / "pointcloud_segment_camera.npz",
        "segment_pointcloud_world_npz": output_dir / "pointcloud_segment_world.npz",
        "full_pointcloud_ply": output_dir / "pointcloud_full_camera.ply",
        "segment_pointcloud_ply": output_dir / "pointcloud_segment_camera.ply",
        "summary": output_dir / "preflight_summary.json",
        "html": output_dir / "preflight_report.html",
    }


def _wait_for_observation(env: Any, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + float(timeout_s)
    last_keys: list[str] = []
    while time.time() < deadline:
        obs = env.get_observation()
        last_keys = sorted(str(k) for k in obs.keys())
        camera = obs.get("robot0_robotview")
        if isinstance(camera, dict):
            images = camera.get("images", {})
            if (
                "rgb" in images
                and "depth" in images
                and "intrinsics" in camera
                and ("pose_mat" in camera or "pose" in camera)
            ):
                return obs
        time.sleep(0.25)
    raise TimeoutError(
        f"Timed out waiting for bridge observation after {timeout_s:.1f}s. "
        f"Last observation keys: {last_keys}"
    )


def _pose_mat_from_obs(camera_obs: dict[str, Any]) -> np.ndarray:
    if "pose_mat" in camera_obs:
        return np.asarray(camera_obs["pose_mat"], dtype=np.float64).reshape(4, 4)
    pose = np.asarray(camera_obs["pose"], dtype=np.float64).reshape(7)
    quat_wxyz = pose[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = SciRotation.from_quat(quat_xyzw).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def _make_low_level_env(config_path: str, observation_port: int | None) -> Any:
    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    configs = DictLoader.load([os.path.expanduser(config_path)])
    low_level_cfg = dict(configs["env"]["cfg"]["low_level"])
    low_level_cfg["dry_run"] = True
    low_level_cfg["wait_for_observation_on_reset"] = False
    if observation_port is not None:
        low_level_cfg["observation_server_port"] = int(observation_port)
    return instantiate(low_level_cfg)


def _pyroki_server_config(configs: dict[str, Any]) -> dict[str, Any]:
    for server_cfg in configs.get("api_servers", []):
        if not isinstance(server_cfg, dict):
            continue
        if server_cfg.get("_target_") == "capx.serving.launch_pyroki_server.main":
            return server_cfg
    return {}


def run(args: argparse.Namespace) -> None:
    from capx.envs.configs.loader import DictLoader
    from capx.envs.runner import _start_api_servers, _stop_api_servers
    from capx.integrations.motion.pyroki import init_pyroki
    from capx.integrations.vision import graspnet, sam3
    from capx.integrations.vision.graspnet import init_contact_graspnet
    from capx.integrations.vision.sam3 import init_sam3
    from capx.utils.depth_utils import depth_to_rgb

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_dir)
    os.environ["CAPX_G1_DRY_RUN"] = "true"
    os.environ["NO_PROXY"] = ",".join(
        sorted(set(filter(None, os.environ.get("NO_PROXY", "").split(","))) | {"127.0.0.1", "localhost"})
    )
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    configs = DictLoader.load([os.path.expanduser(args.config_path)])
    api_servers = configs.get("api_servers", [])
    server_procs: list[Any] = []
    if not args.no_start_api_servers:
        server_procs = _start_api_servers(api_servers, wait_timeout=float(args.api_server_wait_timeout_s))

    env = None
    try:
        sam3.SERVICE_URL = args.sam3_url.rstrip("/")
        graspnet.SERVICE_URL = args.graspnet_url.rstrip("/")

        env = _make_low_level_env(args.config_path, args.observation_port)
        print(f"[g1-preflight] waiting for bridge observation on port {args.observation_port or '<config>'}")
        obs = _wait_for_observation(env, args.observation_timeout_s)
        camera = obs["robot0_robotview"]
        rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
        depth = _normalize_depth(camera["images"]["depth"])
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        t_world_camera = _pose_mat_from_obs(camera)

        Image.fromarray(rgb).save(paths["rgb"])
        np.save(paths["depth"], depth)
        Image.fromarray(depth_to_rgb(depth, use_percentiles=(1, 99))).save(paths["depth_color"])
        np.save(paths["intrinsics"], intrinsics)
        np.save(paths["camera_pose"], t_world_camera)

        print(f"[g1-preflight] running SAM3 prompt={args.object_name!r}")
        sam3_fn = init_sam3(device=args.sam3_device)
        sam3_results = sam3_fn(rgb, text_prompt=args.object_name)
        if not sam3_results:
            raise ValueError("SAM3 returned no detections.")
        sam3_scores = np.asarray([float(item.get("score", 0.0)) for item in sam3_results])
        best_sam3_idx = int(sam3_scores.argmax())
        best = sam3_results[best_sam3_idx]
        mask = _normalize_mask(best["mask"], depth.shape)
        _save_sam3_overlay(rgb, best, paths["sam3_overlay"])
        sam3_summary = {
            "prompt": args.object_name,
            "result_count": len(sam3_results),
            "best_index": best_sam3_idx,
            "results": _summarize_sam3(sam3_results),
        }
        _write_json(paths["sam3_summary"], sam3_summary)

        full_mask = np.ones_like(mask, dtype=bool)
        full_points_cam, full_colors = deproject_masked_points(
            depth,
            rgb,
            full_mask,
            intrinsics,
            stride=max(1, int(args.full_pointcloud_stride)),
        )
        segment_points_cam, segment_colors = deproject_masked_points(
            depth,
            rgb,
            mask,
            intrinsics,
            stride=max(1, int(args.segment_pointcloud_stride)),
        )
        segment_points_world = (
            (t_world_camera[:3, :3] @ segment_points_cam.T).T + t_world_camera[:3, 3]
        )

        np.savez_compressed(paths["full_pointcloud_npz"], points=full_points_cam, colors=full_colors)
        np.savez_compressed(
            paths["segment_pointcloud_npz"],
            points=segment_points_cam,
            colors=segment_colors,
            mask=mask,
        )
        np.savez_compressed(
            paths["segment_pointcloud_world_npz"],
            points=segment_points_world,
            colors=segment_colors,
        )
        _save_pointcloud_ply(paths["full_pointcloud_ply"], full_points_cam, full_colors)
        _save_pointcloud_ply(paths["segment_pointcloud_ply"], segment_points_cam, segment_colors)

        object_pose_cam_mat, obb_summary = _compute_obb_pose(segment_points_cam)
        object_pose_world_mat, object_pose_world = transform_pose_matrix(
            t_world_camera,
            object_pose_cam_mat,
        )
        _, object_pose_camera = transform_pose_matrix(np.eye(4), object_pose_cam_mat)

        print("[g1-preflight] running ContactGraspNet")
        segmap = mask.astype(np.int32)
        grasp_plan_fn = init_contact_graspnet()
        grasps, scores, contact_pts = grasp_plan_fn(
            depth,
            intrinsics,
            segmap,
            1,
        )
        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores_arr.size == 0:
            raise ValueError("ContactGraspNet returned no grasp candidates.")
        best_idx = int(scores_arr.argmax())
        grasp_cam_raw = np.asarray(grasps[best_idx], dtype=np.float64).reshape(4, 4)
        grasp_cam_adjusted = grasp_cam_raw @ _translation_matrix([0.0, 0.0, float(args.grasp_tip_offset_m)])
        grasp_world_mat, grasp_pose_world = transform_pose_matrix(t_world_camera, grasp_cam_adjusted)
        _, grasp_pose_camera = transform_pose_matrix(np.eye(4), grasp_cam_adjusted)

        print("[g1-preflight] solving PyRoKi IK; no joint command will be sent")
        ik_fn = init_pyroki(server_url=args.pyroki_url)
        quat_wxyz = np.asarray(grasp_pose_world["quaternion_wxyz"], dtype=np.float64)
        pos_world = np.asarray(grasp_pose_world["position"], dtype=np.float64)
        rot = SciRotation.from_matrix(grasp_world_mat[:3, :3])
        approach_pos = pos_world + rot.apply(np.array([0.0, 0.0, -float(args.z_approach)]))
        approach_full_cfg = ik_fn(target_pose_wxyz_xyz=np.concatenate([quat_wxyz, approach_pos]))
        final_full_cfg = ik_fn(
            target_pose_wxyz_xyz=np.concatenate([quat_wxyz, pos_world]),
            prev_cfg=np.asarray(approach_full_cfg, dtype=np.float64),
        )
        approach_right_arm = extract_right_arm_ik_solution(approach_full_cfg)
        final_right_arm = extract_right_arm_ik_solution(final_full_cfg)
        pyroki_cfg = _pyroki_server_config(configs)
        final_stick_figure = None
        try:
            final_stick_figure = compute_right_arm_stick_figure(
                robot_urdf_path=str(pyroki_cfg.get("robot", "env_configs/g1/g1_29dof_with_hand.urdf")),
                root_link_name=str(pyroki_cfg.get("root_link", "torso_link")),
                target_link_name=str(pyroki_cfg.get("target_link", "right_hand_palm_link")),
                joints=final_right_arm,
            )
        except Exception as exc:
            print(f"[g1-preflight] WARNING: failed to compute IK stick figure: {exc}", file=sys.stderr)

        artifacts = {name: str(path) for name, path in paths.items()}
        summary = {
            "object_name": args.object_name,
            "timestamp": time.time(),
            "dry_run": True,
            "stopped_after": "pyroki_ik",
            "config_path": args.config_path,
            "services": {
                "sam3_url": args.sam3_url,
                "graspnet_url": args.graspnet_url,
                "pyroki_url": args.pyroki_url,
            },
            "camera": {
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth.shape),
                "intrinsics": intrinsics,
                "T_world_camera": t_world_camera,
            },
            "sam3": sam3_summary,
            "pointcloud": {
                "full_points_saved": int(len(full_points_cam)),
                "segment_points_saved": int(len(segment_points_cam)),
                "segment_world_frame": "torso_link",
            },
            "object_modal_pose_camera": object_pose_camera,
            "object_modal_pose_world": object_pose_world,
            "object_obb_camera": obb_summary,
            "object_modal_pose_world_matrix": object_pose_world_mat,
            "grasp": {
                "candidate_count": int(scores_arr.size),
                "best_index": best_idx,
                "best_score": float(scores_arr[best_idx]),
                "contact_point_camera": np.asarray(contact_pts[best_idx]).tolist()
                if len(np.asarray(contact_pts)) > best_idx
                else None,
                "raw_grasp_pose_camera_matrix": grasp_cam_raw,
                "adjusted_grasp_pose_camera_matrix": grasp_cam_adjusted,
            },
            "grasp_pose_camera": grasp_pose_camera,
            "grasp_pose_world": grasp_pose_world,
            "grasp_pose_world_matrix": grasp_world_mat,
            "ik": {
                "z_approach": float(args.z_approach),
                "approach_position_world": approach_pos,
                "approach_full_cfg": approach_full_cfg,
                "approach_right_arm_joints": approach_right_arm,
                "final_full_cfg": final_full_cfg,
                "final_right_arm_joints": final_right_arm,
                "final_stick_figure": final_stick_figure,
            },
            "artifacts": artifacts,
        }
        _write_json(paths["summary"], summary)
        report_html = build_report_html(
            summary,
            artifacts,
            point_preview=_point_preview(segment_points_world, segment_colors),
        )
        paths["html"].write_text(report_html, encoding="utf-8")

        print(f"[g1-preflight] saved summary: {paths['summary']}")
        print(f"[g1-preflight] saved report:  {paths['html']}")
        print(f"[g1-preflight] final right-arm IK joints: {final_right_arm.tolist()}")
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if server_procs and not args.keep_started_api_servers:
            _stop_api_servers(server_procs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a G1 bottle-grasp dry-run preflight and stop after PyRoKi IK."
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--observation-port", type=int, default=None)
    parser.add_argument("--observation-timeout-s", type=float, default=60.0)
    parser.add_argument("--sam3-url", default=DEFAULT_SAM3_URL)
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--graspnet-url", default=DEFAULT_GRASPNET_URL)
    parser.add_argument("--pyroki-url", default=DEFAULT_PYROKI_URL)
    parser.add_argument("--api-server-wait-timeout-s", type=float, default=180.0)
    parser.add_argument("--no-start-api-servers", action="store_true")
    parser.add_argument("--keep-started-api-servers", action="store_true")
    parser.add_argument("--z-approach", type=float, default=0.10)
    parser.add_argument("--grasp-tip-offset-m", type=float, default=0.12)
    parser.add_argument("--full-pointcloud-stride", type=int, default=2)
    parser.add_argument("--segment-pointcloud-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[g1-preflight] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
