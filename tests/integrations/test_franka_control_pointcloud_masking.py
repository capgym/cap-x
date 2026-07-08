import numpy as np


def test_masked_valid_pointcloud_uses_same_nan_filtered_index_space() -> None:
    from capx.integrations.franka.control import masked_valid_pointcloud

    points = np.arange(18, dtype=np.float64).reshape(6, 3)
    colors = np.arange(18, 36, dtype=np.float64).reshape(6, 3)
    depth = np.array(
        [
            [1.0, np.nan, 2.0],
            [3.0, np.nan, 4.0],
        ],
        dtype=np.float64,
    )
    mask = np.array(
        [
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )

    masked_points, masked_colors = masked_valid_pointcloud(points, colors, depth, mask)

    valid = np.array([True, False, True, True, False, True])
    expected_valid_points = points[valid]
    expected_valid_colors = colors[valid]

    assert np.array_equal(masked_points, expected_valid_points[[2, 3]])
    assert np.array_equal(masked_colors, expected_valid_colors[[2, 3]])


def test_masked_valid_pointcloud_matches_depth_clip_filtering() -> None:
    from capx.integrations.franka.control import masked_valid_pointcloud

    points = np.arange(18, dtype=np.float64).reshape(6, 3)
    colors = np.arange(18, 36, dtype=np.float64).reshape(6, 3)
    depth = np.array(
        [
            [1.0, 0.0, 2.0],
            [3.0, 25.0, 4.0],
        ],
        dtype=np.float64,
    )
    mask = np.array(
        [
            [False, True, False],
            [True, True, True],
        ],
        dtype=bool,
    )

    masked_points, masked_colors = masked_valid_pointcloud(points, colors, depth, mask)

    valid = np.array([True, False, True, True, False, True])
    expected_valid_points = points[valid]
    expected_valid_colors = colors[valid]

    assert np.array_equal(masked_points, expected_valid_points[[2, 3]])
    assert np.array_equal(masked_colors, expected_valid_colors[[2, 3]])


def test_masked_valid_pointcloud_accepts_pre_filtered_pointcloud() -> None:
    from capx.integrations.franka.control import masked_valid_pointcloud

    points = np.arange(12, dtype=np.float64).reshape(4, 3)
    colors = np.arange(12, 24, dtype=np.float64).reshape(4, 3)
    depth = np.array(
        [
            [1.0, np.nan, 2.0],
            [3.0, 25.0, 4.0],
        ],
        dtype=np.float64,
    )
    mask = np.array(
        [
            [False, True, False],
            [True, True, True],
        ],
        dtype=bool,
    )

    masked_points, masked_colors = masked_valid_pointcloud(points, colors, depth, mask)

    assert np.array_equal(masked_points, points[[2, 3]])
    assert np.array_equal(masked_colors, colors[[2, 3]])
