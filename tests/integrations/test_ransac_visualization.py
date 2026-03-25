from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro


@dataclass(frozen=True)
class PlaneModel:
    """Plane model in the form z = a*x + b*y + c."""

    a: float
    b: float
    c: float

    @property
    def normal(self) -> np.ndarray:
        # Implicit form: a*x + b*y - z + c = 0 -> normal [a, b, -1]
        return np.array([self.a, self.b, -1.0])


def fit_plane_least_squares(points_xyz: np.ndarray) -> PlaneModel:
    """Fit z = a*x + b*y + c via least squares.

    Args:
        points_xyz: Array of shape (N, 3) with columns [x, y, z].

    Returns:
        PlaneModel with parameters (a, b, c).
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs.tolist()
    return PlaneModel(a=a, b=b, c=c)


def plane_point_distances(points_xyz: np.ndarray, model: PlaneModel) -> np.ndarray:
    """Compute perpendicular distances from points to plane a*x + b*y - z + c = 0.

    Args:
        points_xyz: Array of shape (N, 3).
        model: PlaneModel.

    Returns:
        Distances of shape (N,).
    """
    a, b, c = model.a, model.b, model.c
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    numer = np.abs(a * x + b * y - z + c)
    denom = np.sqrt(a * a + b * b + 1.0)
    return numer / denom


def ransac(
    points_xyz: np.ndarray,
    inlier_threshold: float = 0.01,
    max_iterations: int = 1000,
    min_points: int = 3,
) -> tuple[PlaneModel, np.ndarray]:
    """RANSAC for fitting a plane z = a*x + b*y + c to 3D points.

    Args:
        points_xyz: Points of shape (N, 3).
        inlier_threshold: Perpendicular distance threshold.
        max_iterations: Number of random trials.
        min_points: Minimal sample size to fit a plane (>=3).

    Returns:
        (best_model, inlier_indices)
    """
    n = points_xyz.shape[0]
    if n < min_points:
        raise ValueError("Not enough points for RANSAC.")

    best_model: PlaneModel | None = None
    best_inliers: np.ndarray = np.array([], dtype=int)

    for _ in range(max_iterations):
        sample_idx = np.random.choice(n, size=min_points, replace=False)
        sample = points_xyz[sample_idx]

        # Fit using least squares to be robust against degenerate triples
        model = fit_plane_least_squares(sample)
        dists = plane_point_distances(points_xyz, model)
        inliers = np.flatnonzero(dists < inlier_threshold)

        if inliers.size > best_inliers.size:
            best_model = model
            best_inliers = inliers

    if best_model is None:
        # Fallback to least squares on all points
        best_model = fit_plane_least_squares(points_xyz)
        best_inliers = np.arange(n)

    # Refit on all inliers for a tighter model
    refined_model = fit_plane_least_squares(points_xyz[best_inliers])
    return refined_model, best_inliers


def visualize_ransac(
    points_xyz: np.ndarray,
    model: PlaneModel,
    inlier_mask: np.ndarray,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Visualize inliers/outliers and fitted plane in 3D.

    Args:
        points_xyz: (N, 3) points.
        model: Fitted plane model.
        inlier_mask: Boolean mask of shape (N,) for inliers.
        save_path: If provided, save figure to this path.
        show: Whether to display window.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter
    ax.scatter(
        x[~inlier_mask],
        y[~inlier_mask],
        z[~inlier_mask],
        c="red",
        s=12,
        label="Outliers",
        alpha=0.7,
    )
    ax.scatter(
        x[inlier_mask],
        y[inlier_mask],
        z[inlier_mask],
        c="limegreen",
        s=14,
        label="Inliers",
        alpha=0.9,
    )

    # Plane surface over data bounds
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 20),
        np.linspace(ylim[0], ylim[1], 20),
    )
    zz = model.a * xx + model.b * yy + model.c
    ax.plot_surface(xx, yy, zz, color="royalblue", alpha=0.25, edgecolor="none")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")
    ax.set_title("RANSAC Plane Fit")
    ax.view_init(elev=22, azim=35)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)


@dataclass
class Config:
    """CLI arguments for RANSAC demo."""

    n_inliers: int = 100
    n_outliers: int = 20
    noise_std: float = 0.1
    inlier_threshold: float = 0.1
    max_iterations: int = 1000
    seed: int = 0
    show: bool = True
    save_path: str | None = None


def main(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)

    # Ground-truth plane z = a*x + b*y + c
    a_true, b_true, c_true = 1.0, 1.0, 0.0

    xy = rng.random((cfg.n_inliers, 2)) * 10.0
    noise = rng.normal(0.0, cfg.noise_std, size=(cfg.n_inliers,))
    z = a_true * xy[:, 0] + b_true * xy[:, 1] + c_true + noise
    inlier_points = np.column_stack([xy, z])

    outliers = rng.random((cfg.n_outliers, 3)) * 10.0
    points = np.vstack([inlier_points, outliers])

    model, inliers_idx = ransac(
        points_xyz=points,
        inlier_threshold=cfg.inlier_threshold,
        max_iterations=cfg.max_iterations,
        min_points=3,
    )

    inlier_mask = np.zeros(points.shape[0], dtype=bool)
    inlier_mask[inliers_idx] = True

    print(
        {
            "model": {"a": model.a, "b": model.b, "c": model.c},
            "num_inliers": int(inlier_mask.sum()),
            "total_points": int(points.shape[0]),
        }
    )

    visualize_ransac(points, model, inlier_mask, save_path=cfg.save_path, show=cfg.show)


if __name__ == "__main__":
    tyro.cli(main)
