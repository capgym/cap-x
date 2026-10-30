"""GraspNet's empty result must retain the matrix shape expected by native consumers."""

from unittest.mock import Mock

import numpy as np
import pytest
import viser.transforms as vtf

from capx.serving import launch_contact_graspnet_server as server


@pytest.mark.parametrize("method", ["_do_plan", "_do_plan_point_clouds", "_do_plan_evenly"])
@pytest.mark.parametrize("count", [0, 1])
def test_grasp_result_shapes(monkeypatch, method, count):
    grasps = np.repeat(np.eye(4)[None], count, axis=0)
    scores = np.ones(count)
    points = np.zeros((count, 3))
    cloud = np.array([[0.0, 0.0, 1.0]])
    estimator = Mock()
    estimator.extract_point_clouds.return_value = (cloud, {1: cloud}, None)
    estimator.predict_scene_grasps.return_value = ({1: grasps}, {1: scores}, {1: points}, None)
    monkeypatch.setattr(server, "_GRASP_ESTIMATOR", estimator)
    monkeypatch.setattr(
        server,
        "sample_random_camera_viewpoint",
        lambda *a, **kw: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(
        server,
        "sample_cone_viewpoints_evenly",
        lambda *a, **kw: (np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]])),
    )
    if method == "_do_plan_point_clouds":
        request = server.PlanPointCloudsRequest(
            pc_full_base64=server._numpy_to_base64(cloud),
            pc_segment_base64=server._numpy_to_base64(cloud),
            max_retries=1,
        )
    else:
        request = server.PlanEvenlyRequest(
            depth_base64=server._numpy_to_base64(np.ones((2, 2))),
            cam_K_base64=server._numpy_to_base64(np.eye(3)),
            segmap_base64=server._numpy_to_base64(np.ones((2, 2), dtype=np.int32)),
            segmap_id=1,
            max_retries=1,
            num_viewpoints=1,
        )
    response = getattr(server, method)(request)
    actual = server._base64_to_numpy(response.grasps_base64)
    assert actual.shape == (count, 4, 4)
    np.testing.assert_allclose(actual, grasps)
    np.testing.assert_array_equal(server._base64_to_numpy(response.scores_base64), scores)
    assert server._base64_to_numpy(response.contact_pts_base64).shape == (count, 3)
    transformed = (
        vtf.SE3.from_matrix(actual) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
    ).as_matrix()
    assert transformed.shape == (count, 4, 4)
