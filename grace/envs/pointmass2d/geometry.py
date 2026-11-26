"""Exact collision geometry of the point-mass task (numpy, float32).

These checks define success/failure in the environment and validity in the
demonstration planner. The agent is a disc of radius ``robot_radius``; an
optional ``clearance`` inflates it further.
"""

import numpy as np


def _reshape_points(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[-1] != 2:
        raise ValueError("Expected points with trailing dimension 2, got {}".format(pts.shape))
    return pts, pts.reshape(-1, 2)


def points_in_collision(points, obstacle_map, robot_radius, clearance=0.0, include_extra=False):
    """True where the agent disc centred at a point overlaps an obstacle or leaves the workspace."""
    pts, flat = _reshape_points(points)
    collision = np.zeros(flat.shape[0], dtype=bool)

    min_limits = obstacle_map.limits[0] + robot_radius + clearance
    max_limits = obstacle_map.limits[1] - robot_radius - clearance
    collision |= np.any(flat < min_limits, axis=1)
    collision |= np.any(flat > max_limits, axis=1)

    sphere_centers = obstacle_map.all_sphere_centers(include_extra=include_extra)
    sphere_radii = obstacle_map.all_sphere_radii(include_extra=include_extra)
    if sphere_centers.size > 0:
        delta = flat[:, None, :] - sphere_centers[None, :, :]
        dist_sq = np.sum(delta * delta, axis=-1)
        radii_sq = (sphere_radii[None, :] + robot_radius + clearance) ** 2
        collision |= np.any(dist_sq <= radii_sq, axis=1)

    box_centers = obstacle_map.all_box_centers(include_extra=include_extra)
    box_sizes = obstacle_map.all_box_sizes(include_extra=include_extra)
    if box_centers.size > 0:
        half_sizes = 0.5 * box_sizes[None, :, :]
        delta = np.abs(flat[:, None, :] - box_centers[None, :, :]) - half_sizes
        outside = np.maximum(delta, 0.0)
        dist_sq = np.sum(outside * outside, axis=-1)
        collision |= np.any(dist_sq <= (robot_radius + clearance) ** 2, axis=1)

    return collision.reshape(pts.shape[:-1])


def segment_in_collision(
    start,
    end,
    obstacle_map,
    robot_radius,
    clearance=0.0,
    include_extra=False,
    step_size=0.01,
):
    """Checks the straight segment start -> end at a spacing of at most ``step_size``."""
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    dist = np.linalg.norm(end - start)
    n_points = max(int(np.ceil(dist / max(step_size, 1e-6))) + 1, 2)
    alphas = np.linspace(0.0, 1.0, n_points, dtype=np.float32)[:, None]
    samples = start[None, :] + (end - start)[None, :] * alphas
    return bool(
        np.any(
            points_in_collision(
                samples,
                obstacle_map=obstacle_map,
                robot_radius=robot_radius,
                clearance=clearance,
                include_extra=include_extra,
            )
        )
    )


def shortcut_path(
    path,
    obstacle_map,
    robot_radius,
    clearance=0.0,
    include_extra=False,
    step_size=0.01,
    n_iterations=128,
    rng=None,
):
    """Random shortcutting of a waypoint path (used by the demonstration planner)."""
    path = np.asarray(path, dtype=np.float32)
    if path.shape[0] <= 2:
        return path

    if rng is None:
        rng = np.random.RandomState()

    shortened = path.copy()
    for _ in range(n_iterations):
        if shortened.shape[0] <= 2:
            break
        i, j = sorted(rng.randint(0, shortened.shape[0], size=2))
        if j <= i + 1:
            continue
        if not segment_in_collision(
            shortened[i],
            shortened[j],
            obstacle_map=obstacle_map,
            robot_radius=robot_radius,
            clearance=clearance,
            include_extra=include_extra,
            step_size=step_size,
        ):
            shortened = np.concatenate([shortened[: i + 1], shortened[j:]], axis=0)
    return shortened


def resample_path_chebyshev(path, max_delta):
    """Resamples a waypoint path so that no step moves more than ``max_delta`` along either axis."""
    path = np.asarray(path, dtype=np.float32)
    if path.shape[0] == 0:
        raise ValueError("Path must contain at least one point")

    resampled = [path[0]]
    for idx in range(path.shape[0] - 1):
        start = path[idx]
        end = path[idx + 1]
        delta = end - start
        n_steps = max(int(np.ceil(np.max(np.abs(delta)) / max(max_delta, 1e-6))), 1)
        for step in range(1, n_steps + 1):
            alpha = float(step) / float(n_steps)
            resampled.append(start + alpha * delta)
    return np.asarray(resampled, dtype=np.float32)


def path_length(positions):
    """Sum of Euclidean step lengths of a sequence of 2D positions."""
    points = np.asarray(positions, dtype=np.float32).reshape(len(positions), -1)[:, :2]
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
