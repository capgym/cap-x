"""Time-parameterize left-arm joint waypoints for the ServoJ command stream."""

import math

import numpy as np
from scipy.interpolate import PchipInterpolator


_MAX_SAMPLES = 300_001
# Critical points of the quintic clock acceleration on [0, 1].
_CLOCK_ACCELERATION_PEAKS = (0.5 - 1 / math.sqrt(12), 0.5 + 1 / math.sqrt(12))


def _progress(time: np.ndarray) -> np.ndarray:
    return time**3 * (10 + time * (-15 + 6 * time))


def _clock_bounds(knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Invert the monotonic quintic at the geometric knots. Bound the clock's
    # derivatives only over the interval where each cubic is actually used.
    lower, upper = np.zeros_like(knots), np.ones_like(knots)
    for _ in range(40):
        middle = (lower + upper) / 2
        before_knot = _progress(middle) < knots
        lower = np.where(before_knot, middle, lower)
        upper = np.where(before_knot, upper, middle)
    start, end = lower[:-1], upper[1:]
    speed_peak = np.clip(0.5, start, end)
    speed = 30 * speed_peak**2 * (1 - speed_peak)**2
    acceleration_peaks = np.clip(np.asarray(_CLOCK_ACCELERATION_PEAKS)[:, None], start, end)
    peak = 60 * acceleration_peaks * (1 - acceleration_peaks) * (1 - 2 * acceleration_peaks)
    return speed[:, None], np.abs(peak).max(axis=0)[:, None]


def _derivative_bounds(curve: PchipInterpolator) -> tuple[np.ndarray, np.ndarray]:
    cubic, quadratic, linear, _ = curve.c
    width = np.diff(curve.x)[:, None]
    # The first derivative is quadratic; check both ends and its stationary point.
    with np.errstate(divide="ignore", over="ignore"):
        turning_point = np.divide(-quadratic, 3 * cubic, out=np.zeros_like(cubic), where=cubic != 0)
    turning_point = np.clip(turning_point, 0, width)
    end_speed = (3 * cubic * width + 2 * quadratic) * width + linear
    turning_speed = (3 * cubic * turning_point + 2 * quadratic) * turning_point + linear
    speed = np.maximum.reduce((np.abs(linear), np.abs(end_speed), np.abs(turning_speed)))
    # The second derivative is linear; its absolute maximum is at an endpoint.
    acceleration = np.maximum(np.abs(2 * quadratic), np.abs(6 * cubic * width + 2 * quadratic))
    return speed, acceleration


def _interval_count(duration_sec: float, rate_hz: int) -> int:
    intervals = duration_sec * rate_hz
    if not math.isfinite(intervals) or intervals > _MAX_SAMPLES - 1:
        raise ValueError(f"ServoJ trajectory requires more than {_MAX_SAMPLES:,} samples")
    return math.ceil(intervals)


def resample_servoj(
    path: np.ndarray,
    duration_sec: float,
    *,
    rate_hz: int = 300,
    max_velocity_rad_s: float = 0.45,
    max_acceleration_rad_s2: float = 1.5,
) -> np.ndarray:
    """Return an (N, 7) joint stream with samples spaced at ``1/rate_hz`` seconds.

    Interpolate the waypoints with shape-preserving cubics along accumulated
    joint travel. A quintic time law comes to rest at the two ends of the
    whole path. Consecutive duplicate waypoints do not insert dwell time.

    The requested duration is a lower bound. Extend it as needed to keep
    each joint within the velocity and acceleration limits. Both endpoints
    are included, and the sampled differences are also checked against the
    limits, including transitions to and from a stationary hold.
    """
    positions = np.asarray(path, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 7 or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("ServoJ path must be a nonempty, finite (N, 7) array")
    if not isinstance(rate_hz, (int, np.integer)) or rate_hz <= 0:
        raise ValueError("ServoJ rate must be a positive integer")
    duration_sec = float(duration_sec)
    if not math.isfinite(duration_sec) or duration_sec < 0:
        raise ValueError("ServoJ duration must be finite and nonnegative")
    if not all(math.isfinite(limit) and limit > 0 for limit in (max_velocity_rad_s, max_acceleration_rad_s2)):
        raise ValueError("ServoJ velocity and acceleration limits must be finite and positive")

    intervals = _interval_count(duration_sec, rate_hz)
    with np.errstate(over="ignore"):
        travel = np.abs(np.diff(positions, axis=0)).max(axis=1)
    if not np.isfinite(travel).all():
        raise ValueError("ServoJ waypoint travel must be finite")
    moving = travel > 0
    if not moving.any():
        return np.repeat(positions[:1], intervals + 1, axis=0)

    # Infinity-norm distance gives each segment progress in proportion to its
    # fastest-changing axis. The floor keeps distinct sub-precision knots ordered.
    distances = travel[moving]
    if len(distances) >= _MAX_SAMPLES:
        raise ValueError(f"ServoJ trajectory requires more than {_MAX_SAMPLES:,} samples")
    weights = np.maximum(distances / distances.max(), 4 * np.finfo(float).eps * len(distances))
    knots = np.r_[0.0, np.cumsum(weights)]
    knots /= knots[-1]
    waypoints = positions[np.r_[True, moving]]
    curve = PchipInterpolator(knots, waypoints, axis=0, extrapolate=False)
    slope, curvature = _derivative_bounds(curve)
    clock_speed, clock_acceleration = _clock_bounds(knots)
    with np.errstate(over="ignore"):
        speed_bound = (clock_speed * slope).max()
        acceleration_bound = (clock_speed**2 * curvature + clock_acceleration * slope).max()
        limit_duration = max(speed_bound / max_velocity_rad_s, np.sqrt(acceleration_bound / max_acceleration_rad_s2))
    intervals = max(len(distances), intervals, _interval_count(limit_duration, rate_hz))

    # Analytical bounds normally suffice. Allow a bounded timing refinement
    # when finite-precision evaluation produces a violation at the sample grid.
    for _ in range(3):
        time = np.linspace(0.0, 1.0, intervals + 1)
        samples = curve(np.clip(_progress(time), 0, 1))
        samples[[0, -1]] = positions[[0, -1]]
        velocity = np.diff(samples, axis=0) * rate_hz
        rest = np.zeros((1, positions.shape[1]))
        acceleration = np.diff(velocity, axis=0, prepend=rest, append=rest) * rate_hz
        ratio = max(np.abs(velocity).max() / max_velocity_rad_s,
                    np.sqrt(np.abs(acceleration).max() / max_acceleration_rad_s2))
        if math.isfinite(ratio) and ratio <= 1 + 1e-9:
            return samples
        intervals = _interval_count(intervals * ratio * (1 + 1e-8) / rate_hz, rate_hz)
    raise ValueError("ServoJ samples could not satisfy the requested velocity and acceleration limits")
