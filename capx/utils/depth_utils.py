import numpy as np
import numpy.typing as npt
from matplotlib import cm


def deproject_pixel_to_camera(
    pixel: tuple[int, int],
    depth: float,
    intrinsics: npt.NDArray[np.float64],
) -> np.ndarray:
    """Deproject a pixel to a camera point.

    Args:
        pixel: Pixel coordinates (x, y)
        depth: Depth image array of shape (H, W)
        intrinsics: Camera intrinsics matrix of shape (3, 3)

    Returns:
        Tuple of (x, y, z) coordinates
    """
    x = (pixel[0] - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (pixel[1] - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z = depth
    return np.array([x, y, z])


def depth_color_to_pointcloud(
    depth: npt.NDArray[np.float64],
    img: npt.NDArray[np.uint8],
    intrinsics: npt.NDArray[np.float64],
    subsample_factor: int = 1,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert depth and rgb image to points.

    Args:
        depth: Depth image array of shape (H, W)
        img: RGB image array of shape (H, W, 3)
        intrinsics: Camera intrinsics matrix of shape (3, 3)
        subsample_factor: Factor to subsample the image (must be > 0)
        depth_clip_range: Tuple of (near, far) clip range for depth values

    Returns:
        Tuple of (points, colors) arrays

    Raises:
        ValueError: If input arrays have incorrect shapes or subsample_factor is invalid
    """
    # Input validation
    if len(depth.shape) != 2:
        raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")

    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Image array must be (H, W, 3), got shape {img.shape}")

    if depth.shape[:2] != img.shape[:2]:
        raise ValueError(
            f"Depth and image dimensions must match: {depth.shape[:2]} vs {img.shape[:2]}"
        )

    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must be (3, 3), got shape {intrinsics.shape}")

    if subsample_factor <= 0:
        raise ValueError(f"Subsample factor must be positive, got {subsample_factor}")
    H, W = depth.shape
    H_subsampled = H // subsample_factor
    W_subsampled = W // subsample_factor
    depth = depth[::subsample_factor, ::subsample_factor]
    img = img[::subsample_factor, ::subsample_factor]

    # Scale intrinsics to match subsampled image
    intrinsics = intrinsics.copy()
    intrinsics[0, 0] /= subsample_factor  # fx
    intrinsics[1, 1] /= subsample_factor  # fy
    intrinsics[0, 2] /= subsample_factor  # cx
    intrinsics[1, 2] /= subsample_factor  # cy

    # Create meshgrid of pixel coordinates using numpy
    w, h = np.meshgrid(np.arange(W_subsampled), np.arange(H_subsampled), indexing="xy")
    pixels = np.stack([w.flatten(), h.flatten()], axis=-1).astype(np.float32)

    # Get z values for all pixels
    z = depth.reshape(-1)

    # Calculate x,y coordinates for all pixels in parallel
    x = (pixels[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (pixels[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1]

    # Stack x,y,z coordinates
    points = np.stack([x, y, z], axis=-1)

    # Get colors for all pixels
    colors = img.reshape(-1, img.shape[-1])[:, :3] / 255.0

    # Filter out NaN and infinite values and depth values outside clipping range
    near_clip, far_clip = depth_clip_range
    valid_mask = (
        ~np.isnan(points).any(axis=1)
        & ~np.isinf(points).any(axis=1)
        & (points[:, 2] <= far_clip)
        & (points[:, 2] >= near_clip)
    )

    return points[valid_mask], colors[valid_mask]


def depth_to_pointcloud(
    depth: npt.NDArray[np.float64],
    intrinsics: npt.NDArray[np.float64],
    subsample_factor: int = 1,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
    filter_invalid: bool = True,
) -> npt.NDArray[np.float64]:
    """Convert depth image to a 3D point cloud in the camera frame.

    Args:
        depth: Depth image array of shape (H, W)
        intrinsics: Camera intrinsics matrix of shape (3, 3)
        subsample_factor: Factor to subsample the image (must be > 0)
        depth_clip_range: Tuple of (near, far) clip range for depth values
        filter_invalid: If True (default), removes points with NaN/inf depth or
            depth outside clip range. If False, returns one point per pixel so the
            output maintains a 1:1 correspondence with the (subsampled) image.

    Returns:
        Points array of shape (N, 3) where N is the number of (valid) points.

    Raises:
        ValueError: If input arrays have incorrect shapes or subsample_factor is invalid
    """
    if len(depth.shape) != 2:
        raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")

    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must be (3, 3), got shape {intrinsics.shape}")

    if subsample_factor <= 0:
        raise ValueError(f"Subsample factor must be positive, got {subsample_factor}")
    H, W = depth.shape
    H_subsampled = H // subsample_factor
    W_subsampled = W // subsample_factor
    depth = depth[::subsample_factor, ::subsample_factor]

    intrinsics = intrinsics.copy()
    intrinsics[0, 0] /= subsample_factor  # fx
    intrinsics[1, 1] /= subsample_factor  # fy
    intrinsics[0, 2] /= subsample_factor  # cx
    intrinsics[1, 2] /= subsample_factor  # cy

    w, h = np.meshgrid(np.arange(W_subsampled), np.arange(H_subsampled), indexing="xy")
    pixels = np.stack([w.flatten(), h.flatten()], axis=-1).astype(np.float32)

    z = depth.reshape(-1)
    x = (pixels[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (pixels[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1]

    points = np.stack([x, y, z], axis=-1)

    if filter_invalid:
        near_clip, far_clip = depth_clip_range
        valid_mask = (
            ~np.isnan(points).any(axis=1)
            & ~np.isinf(points).any(axis=1)
            & (points[:, 2] <= far_clip)
            & (points[:, 2] >= near_clip)
        )
        return points[valid_mask]

    return points


def depth_to_rgb(
    depth: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = "viridis",
    invalid_is_nan: bool = True,
    invalid_color: tuple[int, int, int] = (0, 0, 0),
    use_percentiles: tuple[float, float] | None = None,  # e.g., (2, 98)
    invert: bool = False,
    log_scale: bool = False,
) -> np.ndarray:
    """
    Convert a depth map (H, W) float array to an RGB uint8 image using a colormap.

    Args:
        depth: float32/float64 array with depth (meters, etc.)
        vmin, vmax: explicit min/max for normalization (if None, computed from data/percentiles).
        cmap_name: any matplotlib colormap (e.g., 'magma', 'turbo', 'plasma', 'viridis').
        invalid_is_nan: treat NaNs as invalid mask.
        invalid_color: RGB for invalid pixels.
        use_percentiles: if set, compute vmin/vmax from percentiles (low, high).
        invert: flip colormap direction (near bright/far dark, etc.).
        log_scale: apply log1p before normalization (useful for wide ranges).

    Returns:
        rgb: uint8 array (H, W, 3)
    """
    assert depth.ndim == 2, "depth must be (H, W)"
    d = depth.astype(np.float64).copy()

    # Build invalid mask
    invalid_mask = (np.isnan(d) | np.isinf(d)) if invalid_is_nan else np.zeros_like(d, dtype=bool)
    # You can add custom invalid conditions, e.g., d <= 0:
    # invalid_mask |= (d <= 0)

    # Optionally log-scale (preserves zeros)
    if log_scale:
        # avoid negative/NaN; shift by min positive if needed
        min_pos = np.nanmin(d[d > 0]) if np.any(d > 0) else 1.0
        d = np.log1p(np.maximum(d, min_pos * 1e-9))

    # Determine vmin/vmax
    valid = d[~invalid_mask]
    if valid.size == 0:
        # all invalid; return a solid image
        h, w = d.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[...] = np.array(invalid_color, dtype=np.uint8)
        return rgb

    if use_percentiles is not None:
        low_p, high_p = use_percentiles
        vmin = np.percentile(valid, low_p) if vmin is None else vmin
        vmax = np.percentile(valid, high_p) if vmax is None else vmax
    else:
        vmin = np.min(valid) if vmin is None else vmin
        vmax = np.max(valid) if vmax is None else vmax

    # Avoid divide-by-zero
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Normalize to [0,1]
    norm = (d - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    if invert:
        norm = 1.0 - norm

    # Apply colormap (returns RGBA in [0,1])
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)  # (H, W, 4)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)

    # Paint invalids
    if invalid_is_nan:
        rgb[invalid_mask] = np.array(invalid_color, dtype=np.uint8)

    return rgb
