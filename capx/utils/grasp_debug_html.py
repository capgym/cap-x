from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation


def _prepare_points_and_colors(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    points_all = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite_mask = np.all(np.isfinite(points_all), axis=1)
    points_arr = points_all[finite_mask]

    if colors is None:
        colors_arr = np.tile(np.array([[90, 180, 255]], dtype=np.uint8), (len(points_arr), 1))
    else:
        raw_colors = np.asarray(colors)
        try:
            colors_reshaped = raw_colors.reshape(-1, raw_colors.shape[-1])[:, :3]
        except Exception:
            colors_reshaped = np.empty((0, 3), dtype=np.float64)

        if len(colors_reshaped) == len(points_all):
            colors_arr = colors_reshaped[finite_mask]
        elif len(colors_reshaped) == len(points_arr):
            colors_arr = colors_reshaped
        else:
            colors_arr = np.tile(np.array([[90, 180, 255]], dtype=np.uint8), (len(points_arr), 1))

        colors_arr = np.asarray(colors_arr)
        if colors_arr.dtype.kind == "f" and colors_arr.size and float(np.nanmax(colors_arr)) <= 1.0:
            colors_arr = colors_arr * 255.0
        colors_arr = np.nan_to_num(colors_arr, nan=0.0, posinf=255.0, neginf=0.0)
        colors_arr = np.clip(colors_arr, 0, 255).astype(np.uint8)

    if len(points_arr) > max_points:
        idxs = np.linspace(0, len(points_arr) - 1, max_points, dtype=np.int64)
        points_arr = points_arr[idxs]
        colors_arr = colors_arr[idxs]

    return points_arr, colors_arr


def _pose_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> list[list[float]]:
    position_arr = np.asarray(position, dtype=np.float64).reshape(3)
    quat_arr = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat_xyzw = np.array(
        [quat_arr[1], quat_arr[2], quat_arr[3], quat_arr[0]],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = SciRotation.from_quat(quat_xyzw).as_matrix()
    mat[:3, 3] = position_arr
    return mat.tolist()


def build_grasp_debug_html(
    *,
    object_name: str,
    points: np.ndarray,
    colors: np.ndarray | None,
    grasp_position: np.ndarray,
    grasp_quat_wxyz: np.ndarray,
    approach_position: np.ndarray | None = None,
    max_points: int = 12000,
) -> str:
    """Build a standalone 3D debug page for an object point cloud and grasp pose."""

    point_cloud, point_colors = _prepare_points_and_colors(
        points,
        colors,
        max_points=max(1, int(max_points)),
    )
    grasp_position_arr = np.asarray(grasp_position, dtype=np.float64).reshape(3)
    grasp_quat_arr = np.asarray(grasp_quat_wxyz, dtype=np.float64).reshape(4)
    if approach_position is None:
        approach_arr: np.ndarray | None = None
    else:
        approach_arr = np.asarray(approach_position, dtype=np.float64).reshape(3)

    scene: dict[str, Any] = {
        "objectName": object_name,
        "pointCloud": point_cloud.tolist(),
        "pointColors": point_colors.tolist(),
        "graspPosition": grasp_position_arr.tolist(),
        "graspQuatWxyz": grasp_quat_arr.tolist(),
        "graspPoseMatrix": _pose_matrix(grasp_position_arr, grasp_quat_arr),
        "approachPosition": None if approach_arr is None else approach_arr.tolist(),
    }

    title = f"G1 Grasp Debug: {object_name}"
    template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: #0d1117; color: #e6edf3; overflow: hidden; }
    header { height: 52px; display: flex; align-items: center; gap: 18px; padding: 0 18px; border-bottom: 1px solid #30363d; background: #161b22; }
    h1 { margin: 0; font-size: 16px; font-weight: 650; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; height: calc(100vh - 53px); }
    canvas { width: 100%; height: 100%; display: block; background: #0d1117; }
    aside { border-left: 1px solid #30363d; padding: 14px; background: #161b22; overflow: auto; }
    label { display: grid; gap: 6px; margin: 12px 0; font-size: 12px; color: #8b949e; }
    input { width: 100%; }
    .row { display: flex; justify-content: space-between; gap: 12px; border-top: 1px solid #30363d; padding-top: 10px; margin-top: 10px; font-size: 12px; }
    .value { color: #e6edf3; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; }
    .legend { display: grid; gap: 8px; margin-top: 16px; font-size: 12px; color: #c9d1d9; }
    .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 8px; vertical-align: middle; border-radius: 2px; }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <span id=\"count\"></span>
  </header>
  <main>
    <canvas id=\"view\"></canvas>
    <aside>
      <label>Yaw <input id=\"yaw\" type=\"range\" min=\"-180\" max=\"180\" value=\"-35\"></label>
      <label>Pitch <input id=\"pitch\" type=\"range\" min=\"-89\" max=\"89\" value=\"-20\"></label>
      <label>Roll <input id=\"roll\" type=\"range\" min=\"-180\" max=\"180\" value=\"0\"></label>
      <label>Zoom <input id=\"zoom\" type=\"range\" min=\"30\" max=\"260\" value=\"120\"></label>
      <div class=\"legend\">
        <div><span class=\"swatch\" style=\"background:#ff4d4d\"></span>grasp X axis</div>
        <div><span class=\"swatch\" style=\"background:#3fb950\"></span>grasp Y axis</div>
        <div><span class=\"swatch\" style=\"background:#58a6ff\"></span>grasp Z axis</div>
        <div><span class=\"swatch\" style=\"background:#f2cc60\"></span>world +Z approach point</div>
      </div>
      <div class=\"row\"><span>grasp pos</span><span class=\"value\" id=\"graspPos\"></span></div>
      <div class=\"row\"><span>approach pos</span><span class=\"value\" id=\"approachPos\"></span></div>
    </aside>
  </main>
  <script>
const scene = __SCENE_JSON__;
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const controls = {
  yaw: document.getElementById("yaw"),
  pitch: document.getElementById("pitch"),
  roll: document.getElementById("roll"),
  zoom: document.getElementById("zoom"),
};
document.getElementById("count").textContent = `${scene.pointCloud.length} points`;
document.getElementById("graspPos").textContent = fmt(scene.graspPosition);
document.getElementById("approachPos").textContent = scene.approachPosition ? fmt(scene.approachPosition) : "none";
Object.values(controls).forEach((el) => el.addEventListener("input", draw));
window.addEventListener("resize", draw);

function fmt(v) {
  return v.map((x) => Number(x).toFixed(3)).join(", ");
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

function sceneCenter() {
  if (!scene.pointCloud.length) return scene.graspPosition;
  const c = [0, 0, 0];
  for (const p of scene.pointCloud) {
    c[0] += p[0]; c[1] += p[1]; c[2] += p[2];
  }
  c[0] /= scene.pointCloud.length; c[1] /= scene.pointCloud.length; c[2] /= scene.pointCloud.length;
  return c;
}

function rotate(p, center) {
  let x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
  const yaw = Number(controls.yaw.value) * Math.PI / 180;
  const pitch = Number(controls.pitch.value) * Math.PI / 180;
  const roll = Number(controls.roll.value) * Math.PI / 180;
  let cy = Math.cos(yaw), sy = Math.sin(yaw);
  [x, y] = [cy * x - sy * y, sy * x + cy * y];
  let cp = Math.cos(pitch), sp = Math.sin(pitch);
  [y, z] = [cp * y - sp * z, sp * y + cp * z];
  let cr = Math.cos(roll), sr = Math.sin(roll);
  [x, z] = [cr * x - sr * z, sr * x + cr * z];
  return [x, y, z];
}

function project(p, center, rect) {
  const r = rotate(p, center);
  const scale = Number(controls.zoom.value) * 3.0;
  return [rect.width / 2 + r[0] * scale, rect.height / 2 - r[2] * scale, r[1]];
}

function line3(a, b, center, rect, color, width = 2) {
  const pa = project(a, center, rect);
  const pb = project(b, center, rect);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(pa[0], pa[1]);
  ctx.lineTo(pb[0], pb[1]);
  ctx.stroke();
}

function drawGraspPose(center, rect) {
  const m = scene.graspPoseMatrix;
  const o = scene.graspPosition;
  const axes = [
    { v: [m[0][0], m[1][0], m[2][0]], c: "#ff4d4d" },
    { v: [m[0][1], m[1][1], m[2][1]], c: "#3fb950" },
    { v: [m[0][2], m[1][2], m[2][2]], c: "#58a6ff" },
  ];
  for (const axis of axes) {
    const end = [o[0] + axis.v[0] * 0.08, o[1] + axis.v[1] * 0.08, o[2] + axis.v[2] * 0.08];
    line3(o, end, center, rect, axis.c, 4);
  }
  const po = project(o, center, rect);
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(po[0], po[1], 5, 0, Math.PI * 2);
  ctx.fill();
}

function drawApproach(center, rect) {
  if (!scene.approachPosition) return;
  line3(scene.approachPosition, scene.graspPosition, center, rect, "#f2cc60", 2);
  const p = project(scene.approachPosition, center, rect);
  ctx.fillStyle = "#f2cc60";
  ctx.beginPath();
  ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
  ctx.fill();
}

function draw() {
  const rect = resizeCanvas();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const center = sceneCenter();
  const order = scene.pointCloud.map((p, i) => [project(p, center, rect), i]).sort((a, b) => a[0][2] - b[0][2]);
  for (const [pp, i] of order) {
    const c = scene.pointColors[i] || [90, 180, 255];
    ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    ctx.fillRect(pp[0], pp[1], 2, 2);
  }
  drawApproach(center, rect);
  drawGraspPose(center, rect);
}

draw();
  </script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", html.escape(title))
        .replace("__SCENE_JSON__", json.dumps(scene, separators=(",", ":"), allow_nan=False))
    )


def save_grasp_debug_html(path: str | Path, **kwargs: Any) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_grasp_debug_html(**kwargs), encoding="utf-8")
    return output_path
