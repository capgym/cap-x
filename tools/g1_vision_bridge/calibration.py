from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_transform_mat(
    path: str | None,
    *,
    keys: tuple[str, ...],
    allow_identity: bool = False,
    description: str = "transform",
) -> np.ndarray:
    """Load a 4x4 transform matrix from YAML."""

    if path is None:
        if allow_identity:
            return np.eye(4, dtype=np.float32)
        raise ValueError(f"A {description} YAML is required.")

    with Path(path).expanduser().open("r") as f:
        data = yaml.safe_load(f) or {}

    matrix_value: Any = None
    for key in keys:
        if key in data:
            matrix_value = data[key]
            break
    if matrix_value is None:
        raise ValueError(f"{path} must contain one of: {', '.join(keys)}.")

    pose_mat = np.asarray(matrix_value, dtype=np.float32)
    if pose_mat.shape != (4, 4):
        raise ValueError(f"{description} must have shape 4x4, got {pose_mat.shape}.")
    return np.ascontiguousarray(pose_mat)


def load_pose_mat(path: str | None, *, allow_identity: bool = False) -> np.ndarray:
    """Load camera-to-world pose matrix from a YAML file."""

    if path is None and not allow_identity:
        raise ValueError(
            "A camera extrinsics YAML is required. Pass --extrinsics-yaml or "
            "--allow-identity-extrinsics for bench testing only."
        )
    return load_transform_mat(
        path,
        keys=("T_world_camera", "pose_mat"),
        allow_identity=allow_identity,
        description="Camera extrinsics",
    )


def load_depth_to_rgb_mat(path: str | None) -> np.ndarray | None:
    """Load T_rgb_depth, mapping depth optical-frame points to RGB optical frame."""

    if path is None:
        return None
    return load_transform_mat(
        path,
        keys=("T_rgb_depth", "T_color_depth", "T_depth_to_rgb", "depth_to_rgb_mat"),
        description="Depth-to-RGB extrinsics",
    )


def _matrix_from_yaml_value(value: Any, *, path: str) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        rows = int(value.get("rows", 3))
        cols = int(value.get("cols", 3))
        value = np.asarray(value["data"], dtype=np.float32).reshape(rows, cols)
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"{path} intrinsics must have shape 3x3, got {matrix.shape}.")
    return np.ascontiguousarray(matrix)


def intrinsics_from_values(
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    return np.asarray(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def default_intrinsics_for_image(height: int, width: int) -> np.ndarray:
    focal = float(max(height, width))
    return intrinsics_from_values(
        fx=focal,
        fy=focal,
        cx=(float(width) - 1.0) * 0.5,
        cy=(float(height) - 1.0) * 0.5,
    )


def load_intrinsics(path: str | None) -> np.ndarray | None:
    """Load a 3x3 camera intrinsics matrix from YAML if a path is provided."""

    if path is None:
        return None

    with Path(path).expanduser().open("r") as f:
        data = yaml.safe_load(f) or {}

    for key in ("intrinsics", "K", "camera_matrix", "intrinsics_matrix"):
        if key in data:
            return _matrix_from_yaml_value(data[key], path=path)

    if all(key in data for key in ("fx", "fy", "cx", "cy")):
        return intrinsics_from_values(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
        )

    raise ValueError(
        f"{path} must contain intrinsics/K/camera_matrix/intrinsics_matrix or fx/fy/cx/cy."
    )



def load_distortion_coeffs(path: str | None) -> np.ndarray | None:
    """Load optional distortion coefficients from an intrinsics YAML."""

    if path is None:
        return None
    with Path(path).expanduser().open("r") as f:
        data = yaml.safe_load(f) or {}
    for key in ("distortion_coeffs", "coeffs", "D", "distortion"):
        if key in data:
            coeffs = np.asarray(data[key], dtype=np.float32).reshape(-1)
            return np.ascontiguousarray(coeffs)
    return None
