import asyncio
import base64
import contextlib
import functools
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import tyro
import uvicorn
import viser.transforms as vtf
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from capx.utils.graspnet_utils import camera_so3_looking_at_origin, sample_hemisphere_viewpoint, sample_random_camera_viewpoint, sample_hemisphere_viewpoints_evenly

from capx.utils.graspnet_utils import (
    camera_so3_looking_at_origin,
    sample_cone_viewpoints_evenly,
    sample_hemisphere_viewpoint,
    sample_hemisphere_viewpoints_evenly,
    sample_random_camera_viewpoint,
)

# --- Service Configuration ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
_GRASP_ESTIMATOR: Any | None = None
_DEVICE: str = "cuda"

# Semaphore to serialize GPU access (prevents OOM from concurrent inference)
_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    """Run a blocking GPU function without blocking the event loop."""
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def recursive_key_value_assign(d, ks, v):
    """
    Recursive value assignment to a nested dict
    """
    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]], ks[1:], v)
    elif len(ks) == 1:
        d[ks[0]] = v


def load_contact_graspnet_config(
    checkpoint_dir, batch_size=None, max_epoch=None, data_path=None, arg_configs=None
):
    arg_configs = arg_configs or []

    config_path = os.path.join(checkpoint_dir, "config.yaml")
    # If not found in checkpoint dir, look relative to this file's location
    if not os.path.exists(config_path):
        # Fallback to looking in the vendor directory if we can find it
        pass

    with open(config_path) as f:
        global_config = yaml.safe_load(f)

    for conf in arg_configs:
        k_str, v = conf.split(":")
        with contextlib.suppress(Exception):
            v = eval(v)
        ks = [int(k) if k.isdigit() else k for k in k_str.split(".")]

        recursive_key_value_assign(global_config, ks, v)

    if batch_size is not None:
        global_config["OPTIMIZER"]["batch_size"] = int(batch_size)
    if max_epoch is not None:
        global_config["OPTIMIZER"]["max_epoch"] = int(max_epoch)
    if data_path is not None:
        global_config["DATA"]["data_path"] = data_path

    global_config["DATA"]["classes"] = None

    return global_config


# --- Serialization Helpers ---


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str)
        with io.BytesIO(data) as f:
            return np.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid numpy data: {e}")


# --- API Models ---


class PlanRequest(BaseModel):
    depth_base64: str
    cam_K_base64: str
    segmap_base64: str
    segmap_id: int
    local_regions: bool = True
    filter_grasps: bool = True
    skip_border_objects: bool = False
    z_range: list[float] | None = None
    forward_passes: int = 1
    max_retries: int = 7


class PlanPointCloudsRequest(BaseModel):
    pc_full_base64: str
    pc_segment_base64: str
    segmap_id: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    forward_passes: int = 1
    max_retries: int = 7


class PlanEvenlyRequest(PlanRequest):
    num_viewpoints: int = 10
    max_angle_deg: float = 45.0


class PlanResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    contact_pts_base64: str


def _do_plan(req: PlanRequest) -> PlanResponse:
    """Blocking grasp planning (runs on GPU thread)."""
    depth = _base64_to_numpy(req.depth_base64)
    cam_K = _base64_to_numpy(req.cam_K_base64)
    segmap = _base64_to_numpy(req.segmap_base64)

    # Original logic from grasp_graspnet.py's plan function
    current_retries: int = 0
    max_retries = req.max_retries
    segmap_id = req.segmap_id
    z_range = req.z_range

    if z_range is None:
        z_range = [0.2, 2.0]

    # extract_point_clouds
    pc_full, pc_segments, pc_colors = _GRASP_ESTIMATOR.extract_point_clouds(
        depth,
        cam_K,
        segmap=segmap,
        segmap_id=segmap_id,
        skip_border_objects=req.skip_border_objects,
        z_range=z_range,
    )

    pred_grasps_cam, scores, contact_pts, _ = _GRASP_ESTIMATOR.predict_scene_grasps(
        pc_full,
        pc_segments=pc_segments,
        local_regions=req.local_regions,
        filter_grasps=req.filter_grasps,
        forward_passes=req.forward_passes,
    )

    while len(pred_grasps_cam.get(segmap_id, [])) == 0 and current_retries < max_retries:
        # Determine target point for camera to look at
        if segmap_id in pc_segments and len(pc_segments[segmap_id]) > 0:
            target_centroid = np.mean(pc_segments[segmap_id], axis=0)
        else:
            target_centroid = np.mean(pc_full, axis=0)

        position, wxyz = sample_random_camera_viewpoint(target_centroid, xy_extent_meters=0.25)
        tf_wc = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position]))

        pc_full_h = np.hstack([pc_full, np.ones((pc_full.shape[0], 1))])  # N x 4
        pc_segments_h = {}
        for seg_id in pc_segments:
            pc_segments_h[seg_id] = np.hstack(
                [pc_segments[seg_id], np.ones((pc_segments[seg_id].shape[0], 1))]
            )  # N x 4

        # Transform world -> camera
        tf_cw_matrix = tf_wc.inverse().as_matrix()
        pc_full_cam = (tf_cw_matrix @ pc_full_h.T).T[:, :3]
        pc_segments_cam = {}
        for seg_id in pc_segments:
            pc_segments_cam[seg_id] = (tf_cw_matrix @ pc_segments_h[seg_id].T).T[
                :, :3
            ]

        pred_grasps_cam_new, scores_new, contact_pts_new, _ = (
            _GRASP_ESTIMATOR.predict_scene_grasps(
                pc_full_cam,
                pc_segments=pc_segments_cam,
                local_regions=req.local_regions,
                filter_grasps=req.filter_grasps,
                forward_passes=req.forward_passes,
            )
        )

        # Transform results back to Original Frame
        tf_wc_matrix = tf_wc.as_matrix()
        if segmap_id in pred_grasps_cam_new and len(pred_grasps_cam_new[segmap_id]) > 0:
            # Transform grasps (N, 4, 4)
            grasps_cam = pred_grasps_cam_new[segmap_id]
            grasps_world = np.matmul(tf_wc_matrix, grasps_cam)
            pred_grasps_cam[segmap_id] = grasps_world

            # Transform contact points (N, 3)
            pts_cam = contact_pts_new[segmap_id]
            pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
            pts_world = (tf_wc_matrix @ pts_cam_h.T).T[:, :3]
            contact_pts[segmap_id] = pts_world

            # Scores are invariant
            scores[segmap_id] = scores_new[segmap_id]

        current_retries += 1

    if current_retries >= max_retries:
        grasps_out = np.array([])
        scores_out = np.array([])
        contact_pts_out = np.array([])
    else:
        grasps_result = pred_grasps_cam[segmap_id]
        scores_result = scores[segmap_id]
        contact_pts_result = contact_pts[segmap_id]

        if isinstance(grasps_result, list):
            grasps_out = np.array(grasps_result)
        else:
            grasps_out = grasps_result

        if isinstance(scores_result, list):
            scores_out = np.array(scores_result)
        else:
            scores_out = scores_result

        if isinstance(contact_pts_result, list):
            contact_pts_out = np.array(contact_pts_result)
        else:
            contact_pts_out = contact_pts_result

    return PlanResponse(
        grasps_base64=_numpy_to_base64(grasps_out),
        scores_base64=_numpy_to_base64(scores_out),
        contact_pts_base64=_numpy_to_base64(contact_pts_out),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest):
    if _GRASP_ESTIMATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        return await _run_on_gpu(_do_plan, req)
    except Exception as e:
        logger.error(f"Grasp planning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Grasp planning failed: {e}")


def _do_plan_point_clouds(req: PlanPointCloudsRequest) -> PlanResponse:
    """Blocking grasp planning from pre-computed point clouds (runs on GPU thread)."""
    pc_full = _base64_to_numpy(req.pc_full_base64)
    pc_segment = _base64_to_numpy(req.pc_segment_base64)
    segmap_id = req.segmap_id
    pc_segments = {segmap_id: pc_segment}

    current_retries: int = 0
    max_retries = req.max_retries

    pred_grasps_cam, scores, contact_pts, _ = _GRASP_ESTIMATOR.predict_scene_grasps(
        pc_full,
        pc_segments=pc_segments,
        local_regions=req.local_regions,
        filter_grasps=req.filter_grasps,
        forward_passes=req.forward_passes,
    )

    while len(pred_grasps_cam.get(segmap_id, [])) == 0 and current_retries < max_retries:
        if segmap_id in pc_segments and len(pc_segments[segmap_id]) > 0:
            target_centroid = np.mean(pc_segments[segmap_id], axis=0)
        else:
            target_centroid = np.mean(pc_full, axis=0)

        position, wxyz = sample_random_camera_viewpoint(target_centroid, xy_extent_meters=0.25)
        tf_wc = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position]))

        pc_full_h = np.hstack([pc_full, np.ones((pc_full.shape[0], 1))])
        pc_segments_h = {}
        for seg_id in pc_segments:
            pc_segments_h[seg_id] = np.hstack(
                [pc_segments[seg_id], np.ones((pc_segments[seg_id].shape[0], 1))]
            )

        tf_cw_matrix = tf_wc.inverse().as_matrix()
        pc_full_cam = (tf_cw_matrix @ pc_full_h.T).T[:, :3]
        pc_segments_cam = {}
        for seg_id in pc_segments:
            pc_segments_cam[seg_id] = (tf_cw_matrix @ pc_segments_h[seg_id].T).T[:, :3]

        pred_grasps_cam_new, scores_new, contact_pts_new, _ = (
            _GRASP_ESTIMATOR.predict_scene_grasps(
                pc_full_cam,
                pc_segments=pc_segments_cam,
                local_regions=req.local_regions,
                filter_grasps=req.filter_grasps,
                forward_passes=req.forward_passes,
            )
        )

        tf_wc_matrix = tf_wc.as_matrix()
        if segmap_id in pred_grasps_cam_new and len(pred_grasps_cam_new[segmap_id]) > 0:
            grasps_cam = pred_grasps_cam_new[segmap_id]
            grasps_world = np.matmul(tf_wc_matrix, grasps_cam)
            pred_grasps_cam[segmap_id] = grasps_world

            pts_cam = contact_pts_new[segmap_id]
            pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
            pts_world = (tf_wc_matrix @ pts_cam_h.T).T[:, :3]
            contact_pts[segmap_id] = pts_world

            scores[segmap_id] = scores_new[segmap_id]

        current_retries += 1

    if current_retries >= max_retries:
        grasps_out = np.array([])
        scores_out = np.array([])
        contact_pts_out = np.array([])
    else:
        grasps_result = pred_grasps_cam[segmap_id]
        scores_result = scores[segmap_id]
        contact_pts_result = contact_pts[segmap_id]

        grasps_out = np.array(grasps_result) if isinstance(grasps_result, list) else grasps_result
        scores_out = (
            np.array(scores_result) if isinstance(scores_result, list) else scores_result
        )
        contact_pts_out = (
            np.array(contact_pts_result)
            if isinstance(contact_pts_result, list)
            else contact_pts_result
        )

    return PlanResponse(
        grasps_base64=_numpy_to_base64(grasps_out),
        scores_base64=_numpy_to_base64(scores_out),
        contact_pts_base64=_numpy_to_base64(contact_pts_out),
    )


@app.post("/plan_point_clouds", response_model=PlanResponse)
async def plan_point_clouds_endpoint(req: PlanPointCloudsRequest):
    if _GRASP_ESTIMATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        return await _run_on_gpu(_do_plan_point_clouds, req)
    except Exception as e:
        logger.error(f"Point cloud grasp planning failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Point cloud grasp planning failed: {e}"
        )


def _do_plan_evenly(req: PlanEvenlyRequest) -> PlanResponse:
    """Blocking evenly-sampled grasp planning (runs on GPU thread)."""
    depth = _base64_to_numpy(req.depth_base64)
    cam_K = _base64_to_numpy(req.cam_K_base64)
    segmap = _base64_to_numpy(req.segmap_base64)

    segmap_id = req.segmap_id
    z_range = req.z_range

    if z_range is None:
        z_range = [0.2, 2.0]

    # extract_point_clouds
    pc_full, pc_segments, pc_colors = _GRASP_ESTIMATOR.extract_point_clouds(
        depth,
        cam_K,
        segmap=segmap,
        segmap_id=segmap_id,
        skip_border_objects=req.skip_border_objects,
        z_range=z_range,
    )

    # Determine target point for camera to look at
    if segmap_id in pc_segments and len(pc_segments[segmap_id]) > 0:
        target_centroid = np.mean(pc_segments[segmap_id], axis=0)
    else:
        target_centroid = np.mean(pc_full, axis=0)

    # Sample N viewpoints evenly
    positions, wxyzs = sample_cone_viewpoints_evenly(
        target_centroid,
        current_camera_position=np.array([0.0, 0.0, 0.0]),
        num_samples=req.num_viewpoints,
        max_angle_deg=req.max_angle_deg,
    )

    # We will accumulate results across all viewpoints
    aggregated_grasps = []
    aggregated_scores = []
    aggregated_contact_pts = []

    # Prepare homogenous coords for transformation
    pc_full_h = np.hstack([pc_full, np.ones((pc_full.shape[0], 1))])
    pc_segments_h = {}
    for seg_id in pc_segments:
        pc_segments_h[seg_id] = np.hstack(
            [pc_segments[seg_id], np.ones((pc_segments[seg_id].shape[0], 1))]
        )

    for i in range(req.num_viewpoints):
        position = positions[i]
        wxyz = wxyzs[i]

        # Camera pose in World (Original) frame
        tf_wc = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position]))

        # Transform World -> Camera
        tf_cw_matrix = tf_wc.inverse().as_matrix()

        pc_full_cam = (tf_cw_matrix @ pc_full_h.T).T[:, :3]
        pc_segments_cam = {}
        for seg_id in pc_segments:
            pc_segments_cam[seg_id] = (tf_cw_matrix @ pc_segments_h[seg_id].T).T[:, :3]

        # Predict grasps in this camera frame
        pred_grasps_cam, scores, contact_pts, _ = _GRASP_ESTIMATOR.predict_scene_grasps(
            pc_full_cam,
            pc_segments=pc_segments_cam,
            local_regions=req.local_regions,
            filter_grasps=req.filter_grasps,
            forward_passes=req.forward_passes,
        )

        if segmap_id in pred_grasps_cam and len(pred_grasps_cam[segmap_id]) > 0:
            # Transform grasps back to World frame
            tf_wc_matrix = tf_wc.as_matrix()

            # Grasps: (N, 4, 4)
            grasps_cam = pred_grasps_cam[segmap_id]
            grasps_world = np.matmul(tf_wc_matrix, grasps_cam)

            # Contact points: (N, 3)
            pts_cam = contact_pts[segmap_id]
            pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
            pts_world = (tf_wc_matrix @ pts_cam_h.T).T[:, :3]

            scores_val = scores[segmap_id]

            aggregated_grasps.append(grasps_world)
            aggregated_scores.append(scores_val)
            aggregated_contact_pts.append(pts_world)

    # Concatenate all results
    if len(aggregated_grasps) > 0:
        grasps_out = np.concatenate(aggregated_grasps, axis=0)
        scores_out = np.concatenate(aggregated_scores, axis=0)
        contact_pts_out = np.concatenate(aggregated_contact_pts, axis=0)
    else:
        grasps_out = np.array([])
        scores_out = np.array([])
        contact_pts_out = np.array([])

    return PlanResponse(
        grasps_base64=_numpy_to_base64(grasps_out),
        scores_base64=_numpy_to_base64(scores_out),
        contact_pts_base64=_numpy_to_base64(contact_pts_out),
    )


@app.post("/plan_evenly", response_model=PlanResponse)
async def plan_evenly_endpoint(req: PlanEvenlyRequest):
    if _GRASP_ESTIMATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        return await _run_on_gpu(_do_plan_evenly, req)
    except Exception as e:
        logger.error(f"Evenly-sampled grasp planning failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Evenly-sampled grasp planning failed: {e}"
        )


def main(device: str = "cuda", port: int = 8115, host: str = "127.0.0.1"):
    global _GRASP_ESTIMATOR, _DEVICE
    _DEVICE = device

    # --- Setup Paths & Import ---

    here = os.path.dirname(os.path.abspath(__file__))
    # Assume capx/serving -> go up to hyrl -> go to third_party
    vendor_root = os.path.normpath(
        os.path.join(here, "..", "third_party", "contact_graspnet_pytorch")
    )

    pointnet_root = os.path.join(vendor_root, "Pointnet_Pointnet2_pytorch")
    if pointnet_root not in sys.path:
        sys.path.append(pointnet_root)
    sys.path.append(vendor_root)

    try:
        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    except ImportError:
        logger.error(
            f"Could not import contact_graspnet_pytorch. Verified vendor_root: {vendor_root}"
        )
        raise

    model_checkpoint_dir = os.path.join(vendor_root, "checkpoints/contact_graspnet/checkpoints")

    logger.info(f"Loading GraspNet config from {model_checkpoint_dir}")
    global_config = load_contact_graspnet_config(Path(model_checkpoint_dir).parent)

    logger.info("Building GraspNet model...")
    _GRASP_ESTIMATOR = GraspEstimator(global_config)

    logger.info(f"Loading weights from {model_checkpoint_dir}")
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=_GRASP_ESTIMATOR.model)
    try:
        checkpoint_io.load("model.pt")
    except FileExistsError:
        logger.warning("No model checkpoint found")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")

    logger.info(f"GraspNet Service initialized on {device}. Starting Uvicorn...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)