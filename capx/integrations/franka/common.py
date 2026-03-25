"""Shared utilities for Franka control APIs.

This module centralizes code patterns that were duplicated across many API files:
- TCP offset application
- IK convergence loops
- Gripper control (open/close with stepping)
- Segmentation map construction from SAM2 masks
- Segmentation debug visualization
- Bounding box computation helpers
- Oriented bounding box from 3D points
- Drawing bounding boxes on images
- Dual-arm IK and gripper helpers
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

# Default TCP offset shared across most Franka APIs
DEFAULT_TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)


# ---------------------------------------------------------------------------
# TCP offset helpers
# ---------------------------------------------------------------------------

def apply_tcp_offset(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    tcp_offset: np.ndarray,
) -> np.ndarray:
    """Apply a TCP offset to a target position given orientation.

    Args:
        pos: (3,) target position.
        quat_wxyz: (4,) orientation as [w, x, y, z].
        tcp_offset: (3,) offset in the end-effector frame.

    Returns:
        (3,) offset position.
    """
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    rot = SciRotation.from_quat(quat_xyzw)
    return pos + rot.apply(tcp_offset)


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] to [x, y, z, w]."""
    return np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# IK convergence loop
# ---------------------------------------------------------------------------

def solve_ik_with_convergence(
    ik_solve_fn,
    target_quat_wxyz: np.ndarray,
    target_offset_pos: np.ndarray,
    prev_cfg: np.ndarray | None,
    max_iters: int = 5,
) -> np.ndarray:
    """Run iterative IK solving until convergence.

    Args:
        ik_solve_fn: Callable IK solver.
        target_quat_wxyz: (4,) target orientation [w, x, y, z].
        target_offset_pos: (3,) target position with TCP offset already applied.
        prev_cfg: Previous joint configuration for warm-starting.
        max_iters: Maximum iterations.

    Returns:
        IK solution configuration (including gripper as last element).
    """
    cfg = prev_cfg
    for _ in range(max_iters):
        cfg = ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([target_quat_wxyz, target_offset_pos]),
            prev_cfg=cfg,
        )
        if prev_cfg is not None:
            if np.allclose(cfg, prev_cfg, atol=1e-3):
                break
            else:
                prev_cfg = cfg
        else:
            prev_cfg = cfg
    return cfg


def extract_arm_joints(cfg: np.ndarray) -> np.ndarray:
    """Extract 7-DOF arm joints from a configuration (strip gripper)."""
    return np.asarray(cfg[:-1], dtype=np.float64).reshape(7)


# ---------------------------------------------------------------------------
# Gripper helpers
# ---------------------------------------------------------------------------

def open_gripper(env, steps: int = 30) -> None:
    """Open gripper fully with stepping."""
    env._set_gripper(1.0)
    for _ in range(steps):
        env._step_once()


def close_gripper(env, steps: int = 30) -> None:
    """Close gripper fully with stepping."""
    env._set_gripper(0.0)
    for _ in range(steps):
        env._step_once()


def open_gripper_arm1(env, steps: int = 30) -> None:
    """Open gripper fully for arm 1 with stepping."""
    if not hasattr(env, "_set_gripper_arm1"):
        raise RuntimeError("Environment does not support Arm 1 control")
    env._set_gripper_arm1(1.0)
    for _ in range(steps):
        env._step_once()


def close_gripper_arm1(env, steps: int = 30) -> None:
    """Close gripper fully for arm 1 with stepping."""
    if not hasattr(env, "_set_gripper_arm1"):
        raise RuntimeError("Environment does not support Arm 1 control")
    env._set_gripper_arm1(0.0)
    for _ in range(steps):
        env._step_once()


# ---------------------------------------------------------------------------
# Segmentation helpers (SAM2-based)
# ---------------------------------------------------------------------------

def build_segmentation_map_from_sam2(
    sam2_seg_fn,
    rgb: np.ndarray,
    obs_images: dict[str, Any],
    box: list[float] | None = None,
) -> np.ndarray:
    """Build an integer segmentation map from SAM2 masks.

    First checks if the observation already provides a ``segmentation`` image.
    Falls back to SAM2 inference (with optional box prompt, then global).

    Args:
        sam2_seg_fn: SAM2 segmentation callable.
        rgb: (H, W, 3) uint8 RGB image.
        obs_images: The ``images`` sub-dict from an observation camera entry.
        box: Optional [x1, y1, x2, y2] bounding box.

    Returns:
        (H, W, 1) int32 segmentation map.
    """
    segmentation = obs_images.get("segmentation")
    if segmentation is not None:
        if segmentation.ndim == 2:
            segmentation = segmentation[..., None]
        return segmentation.astype(np.int32, copy=False)

    print("Running SAM2 segmentation with box:", box)

    masks = sam2_seg_fn(rgb, box=box)
    if len(masks) == 0:
        raise RuntimeError("SAM2 returned no masks while attempting to segment scene.")

    if box is not None:
        max_score = -1
        max_idx = -1
        for idx, entry in enumerate(masks, start=1):
            score = entry.get("score")
            if score is not None and score > max_score:
                max_score = score
                max_idx = idx
        masks = [masks[max_idx]]

    seg_map = _masks_to_seg_map(masks, rgb.shape[:2])

    if seg_map.max() == 0:
        print("No masks found with box, Running SAM2 segmentation with global method")
        masks = sam2_seg_fn(rgb)
        if len(masks) == 0:
            raise RuntimeError("SAM2 returned no masks while attempting to segment scene.")
        seg_map = _masks_to_seg_map(masks, rgb.shape[:2])

    if seg_map.max() == 0:
        raise RuntimeError("SAM2 masks were empty; cannot build segmentation map.")
    return seg_map


def _masks_to_seg_map(masks: list, shape: tuple[int, int]) -> np.ndarray:
    """Convert a list of mask dicts to a single integer segmentation map."""
    height, width = shape
    seg_map = np.zeros((height, width, 1), dtype=np.int32)
    for idx, entry in enumerate(masks, start=1):
        mask_obj = entry.get("mask") if isinstance(entry, dict) else None
        if mask_obj is None and hasattr(entry, "mask"):
            mask_obj = entry.mask
        if mask_obj is None:
            continue
        mask = np.asarray(mask_obj, dtype=bool)
        if mask.shape != (height, width):
            try:
                mask = mask.reshape(height, width)
            except ValueError:
                continue
        if mask.any():
            seg_map[mask, 0] = idx
    return seg_map


# ---------------------------------------------------------------------------
# Debug visualization helpers
# ---------------------------------------------------------------------------

def save_segmentation_debug(segmentation: np.ndarray, path: pathlib.Path) -> None:
    """Save a segmentation map as a color-coded debug image with ID labels."""
    denom = float(segmentation.max()) if segmentation.max() > 0 else 1.0
    vis = ((np.repeat(segmentation, 3, axis=2) / denom) * 255.0).astype(np.uint8)
    img = Image.fromarray(vis)

    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        img.save(path)
        return

    draw = ImageDraw.Draw(img)
    height, width = segmentation.shape[:2]
    font_size = max(int(min(height, width) * 0.04), 12)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    unique_vals = np.unique(segmentation)
    for val in unique_vals:
        if val <= 0:
            continue
        mask = segmentation[:, :, 0] == val if segmentation.ndim == 3 else segmentation == val
        if not np.any(mask):
            continue
        ys, xs = np.nonzero(mask)
        cy = float(ys.mean())
        cx = float(xs.mean())
        draw.text(
            (cx, cy),
            str(int(val)),
            fill=(255, 0, 0),
            anchor="mm",
            font=font,
        )

    img.save(path)


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def compute_bbox_indices(
    box: list[float], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Clip and convert a float bounding box to integer pixel indices.

    Args:
        box: [x1, y1, x2, y2] in pixel coordinates.
        shape: (height, width).

    Returns:
        (x1, x2, y1, y2) clipped integer indices.
    """
    height, width = shape
    x1 = int(np.clip(np.floor(box[0]), 0, width - 1))
    y1 = int(np.clip(np.floor(box[1]), 0, height - 1))
    x2 = int(np.clip(np.ceil(box[2]), x1 + 1, width))
    y2 = int(np.clip(np.ceil(box[3]), y1 + 1, height))
    return x1, x2, y1, y2


def select_instance_from_box(
    segmentation: np.ndarray, box: list[float]
) -> tuple[int, np.ndarray]:
    """Select the dominant instance ID within a bounding box.

    Args:
        segmentation: (H, W, 1) or (H, W) int segmentation map.
        box: [x1, y1, x2, y2].

    Returns:
        (instance_id, seg_crop) where seg_crop is the cropped segmentation.
    """
    height, width = segmentation.shape[:2]
    x1, x2, y1, y2 = compute_bbox_indices(box, (height, width))
    seg_crop = segmentation[y1:y2, x1:x2]
    unique_vals, counts = np.unique(seg_crop, return_counts=True)
    valid_mask = unique_vals > 0
    if not np.any(valid_mask):
        raise RuntimeError("No segmented instance overlaps detection bounding box.")
    unique_vals = unique_vals[valid_mask]
    counts = counts[valid_mask]
    queried_instance_idx = int(unique_vals[np.argmax(counts)])
    return queried_instance_idx, seg_crop


# ---------------------------------------------------------------------------
# Oriented bounding box
# ---------------------------------------------------------------------------

def get_oriented_bounding_box_from_3d_points(points: np.ndarray) -> dict[str, Any]:
    """Compute the oriented bounding box from 3D points.

    Adds small noise to avoid degenerate cases, removes outliers, then computes OBB.

    Args:
        points: (N, 3) float64 points.

    Returns:
        dict with "center", "extent", "R" keys.
    """
    points = points + np.random.normal(0, 0.0001, points.shape)
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(points)
    o3d_points, _ = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    obb = o3d_points.get_oriented_bounding_box()
    return {
        "center": obb.center,
        "extent": obb.extent,
        "R": obb.R,
    }


# ---------------------------------------------------------------------------
# Dual-arm frame transform
# ---------------------------------------------------------------------------

def transform_pose_arm0_to_arm1(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    env,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a pose from robot0's base frame to robot1's base frame.

    Args:
        position: (3,) position in robot0 frame.
        quaternion_wxyz: (4,) [w,x,y,z] quaternion in robot0 frame.
        env: Environment with base_link_wxyz_xyz_0 and base_link_wxyz_xyz_1 attributes.

    Returns:
        (pos, quat_wxyz) in robot1's base frame.
    """
    import viser.transforms as vtf

    if not hasattr(env, "base_link_wxyz_xyz_0") or not hasattr(env, "base_link_wxyz_xyz_1"):
        raise RuntimeError(
            "Environment does not provide base transforms. "
            "Make sure you're using a two-arm environment."
        )

    pose_arm0_base = vtf.SE3.from_rotation_and_translation(
        rotation=vtf.SO3(wxyz=quaternion_wxyz),
        translation=position,
    )
    base0_transform = vtf.SE3(wxyz_xyz=env.base_link_wxyz_xyz_0)
    pose_world = base0_transform @ pose_arm0_base

    base1_transform = vtf.SE3(wxyz_xyz=env.base_link_wxyz_xyz_1)
    pose_arm1_base = base1_transform.inverse() @ pose_world

    pos = np.asarray(pose_arm1_base.translation(), dtype=np.float64).reshape(3)
    quat_wxyz = np.asarray(pose_arm1_base.rotation().wxyz, dtype=np.float64).reshape(4)
    return pos, quat_wxyz


# ---------------------------------------------------------------------------
# Drawing utility
# ---------------------------------------------------------------------------

def draw_boxes(
    rgb: np.ndarray,
    boxes: list[list[float]],
    labels: list[str],
    scores: list[float] | None = None,
) -> Image.Image:
    """Draw bounding boxes with labels on an RGB image.

    Args:
        rgb: (H, W, 3) uint8 image.
        boxes: List of [x1, y1, x2, y2].
        labels: List of label strings.
        scores: Optional list of confidence scores.

    Returns:
        PIL Image with drawn boxes.
    """
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for b, lab in zip(boxes, labels, strict=False):
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 12)), lab, fill=(255, 0, 0))
    if scores is not None:
        for b, score in zip(boxes, scores, strict=False):
            x1, y1, x2, y2 = b
            draw.text((x1 + 100, max(0, y1 - 12)), f"{score:.2f}", fill=(255, 0, 0))
    return img
