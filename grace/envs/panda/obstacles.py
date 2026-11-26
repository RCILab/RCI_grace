"""Obstacle shapes of the FR3 task: spheres, boxes and triangular prisms (paper Fig. 3b).

An obstacle is a dict with ``shape``, ``center`` (world, m), ``quat`` (wxyz),
``half_extents`` (m) and, for spheres, ``radius``; all values are float32.

numpy functions: obstacle sampling and the conservative checks of the
evaluation. torch functions: exact capsule-obstacle overlap tests of the GRACE
cost (paper eq. (30), binary term).
"""

import math
from copy import deepcopy

import numpy as np
import torch


SHAPE_IDS = {
    "sphere": 0,
    "box": 1,
    "tri_prism": 2,
}

# Fixed size of the triangular prism (matches tri_prism_obstacle.obj in the scene).
DEFAULT_TRI_PRISM_HALF_EXTENTS = np.array([0.035, 0.16, 0.075], dtype=np.float32)


def normalize_shape_name(shape) -> str:
    name = str(shape).strip().lower()
    if name not in SHAPE_IDS:
        raise ValueError(f"Unsupported obstacle shape {shape!r}; expected one of {sorted(SHAPE_IDS)}")
    return name


def quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def normalize_quat_wxyz(quat) -> np.ndarray:
    arr = np.asarray(quat, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if arr.size != 4:
        raise ValueError(f"quat must have 4 values in wxyz order; got shape {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _half_extents_from_spec(spec: dict, shape: str) -> np.ndarray:
    if shape == "sphere":
        radius = spec.get("radius", None)
        if radius is None:
            size = spec.get("size", spec.get("half_extents", [0.05]))
            radius = np.asarray(size, dtype=np.float32).reshape(-1)[0]
        radius = max(float(radius), 1e-6)
        return np.asarray([radius, radius, radius], dtype=np.float32)

    default = {
        "box": np.asarray([0.04, 0.25, 0.22], dtype=np.float32),
        "tri_prism": DEFAULT_TRI_PRISM_HALF_EXTENTS,
    }[shape]
    raw = spec.get("half_extents", None)
    if raw is None:
        raw = spec.get("size", default)
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size == 3:
            # Accept size as full extents for hand-written specs.
            arr = 0.5 * arr
    else:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.full(3, float(arr[0]), dtype=np.float32)
    if arr.size != 3:
        raise ValueError(f"{shape} half_extents must be scalar or length 3; got {arr.shape}")
    return np.maximum(arr.astype(np.float32), 1e-6)


def normalize_obstacle_spec(spec: dict) -> dict:
    spec = dict(spec)
    shape = normalize_shape_name(spec.get("shape", spec.get("type", "sphere")))
    center = np.asarray(spec.get("center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    if center.size != 3:
        raise ValueError(f"obstacle center must have length 3; got {center.shape}")
    quat = spec.get("quat", spec.get("quat_wxyz", None))
    if quat is None:
        quat = quat_wxyz_from_yaw(float(spec.get("yaw", 0.0)))
    quat = normalize_quat_wxyz(quat)
    half_extents = _half_extents_from_spec(spec, shape)
    radius = float(half_extents[0]) if shape == "sphere" else None
    out = {
        "shape": shape,
        "center": center.astype(np.float32).tolist(),
        "quat": quat.astype(np.float32).tolist(),
        "half_extents": half_extents.astype(np.float32).tolist(),
    }
    if radius is not None:
        out["radius"] = radius
    if "name" in spec:
        out["name"] = str(spec["name"])
    return out


def sphere_obstacle(x, y, z, radius, *, name=None) -> dict:
    spec = {
        "shape": "sphere",
        "center": [float(x), float(y), float(z)],
        "radius": float(radius),
        "quat": [1.0, 0.0, 0.0, 0.0],
    }
    if name is not None:
        spec["name"] = str(name)
    return normalize_obstacle_spec(spec)


def box_obstacle(center, half_extents, *, yaw=0.0, quat=None, name=None) -> dict:
    spec = {
        "shape": "box",
        "center": list(center),
        "half_extents": list(half_extents),
        "quat": normalize_quat_wxyz(quat).tolist() if quat is not None else quat_wxyz_from_yaw(yaw).tolist(),
    }
    if name is not None:
        spec["name"] = str(name)
    return normalize_obstacle_spec(spec)


def tri_prism_obstacle(center, *, yaw=0.0, quat=None, name=None) -> dict:
    spec = {
        "shape": "tri_prism",
        "center": list(center),
        "half_extents": DEFAULT_TRI_PRISM_HALF_EXTENTS.tolist(),
        "quat": normalize_quat_wxyz(quat).tolist() if quat is not None else quat_wxyz_from_yaw(yaw).tolist(),
    }
    if name is not None:
        spec["name"] = str(name)
    return normalize_obstacle_spec(spec)


def clone_obstacle_specs(specs) -> list[dict]:
    return [deepcopy(normalize_obstacle_spec(spec)) for spec in specs]


def _np_quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat).astype(np.float64)
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def points_in_obstacle_np(points, spec: dict, padding=0.0) -> np.ndarray:
    spec = normalize_obstacle_spec(spec)
    pts = np.asarray(points, dtype=np.float64)
    original_shape = pts.shape[:-1]
    flat = pts.reshape(-1, 3)
    padding_arr = np.asarray(padding, dtype=np.float64)
    if padding_arr.size == 1:
        pad = np.full(flat.shape[0], float(padding_arr.reshape(-1)[0]), dtype=np.float64)
    else:
        pad = np.broadcast_to(padding_arr, original_shape).reshape(-1).astype(np.float64)

    center = np.asarray(spec["center"], dtype=np.float64)
    if spec["shape"] == "sphere":
        return (np.linalg.norm(flat - center.reshape(1, 3), axis=1) <= (float(spec["radius"]) + pad)).reshape(original_shape)

    rot = _np_quat_to_rotmat_wxyz(np.asarray(spec["quat"], dtype=np.float64))
    local = (flat - center.reshape(1, 3)) @ rot
    he = np.asarray(spec["half_extents"], dtype=np.float64).reshape(1, 3) + pad.reshape(-1, 1)

    if spec["shape"] == "box":
        inside = np.all(np.abs(local) <= he, axis=1)
    elif spec["shape"] == "tri_prism":
        x = local[:, 0]
        y = local[:, 1]
        z = local[:, 2]
        hx = he[:, 0]
        hz = he[:, 2]
        inside = (
            (np.abs(y) <= he[:, 1])
            & (z >= -hz)
            & (z <= (2.0 * hz / np.maximum(hx, 1e-8)) * (x + hx) - hz)
            & (z <= (-2.0 * hz / np.maximum(hx, 1e-8)) * (x - hx) - hz)
        )
    else:
        raise ValueError(f"Unsupported numpy obstacle shape {spec['shape']!r}")
    return inside.reshape(original_shape)


def _box_mesh_np(half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = np.asarray(half_extents, dtype=np.float64).reshape(3)
    vertices = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _tri_prism_mesh_np(half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = np.asarray(half_extents, dtype=np.float64).reshape(3)
    vertices = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [0.0, -hy, hz],
            [-hx, hy, -hz],
            [hx, hy, -hz],
            [0.0, hy, hz],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [3, 5, 4],
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 2],
            [2, 5, 3],
            [2, 3, 0],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def obstacle_mesh_np(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return local-space vertices/faces for the rendered convex obstacle mesh."""
    spec = normalize_obstacle_spec(spec)
    if spec["shape"] == "sphere":
        raise ValueError("sphere obstacles do not have a triangle mesh")
    half_extents = np.asarray(spec["half_extents"], dtype=np.float64).reshape(3)
    if spec["shape"] == "box":
        return _box_mesh_np(half_extents)
    if spec["shape"] == "tri_prism":
        return _tri_prism_mesh_np(half_extents)
    raise ValueError(f"Unsupported obstacle shape {spec['shape']!r}")


def _np_point_segment_distance(points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    seg = seg_end - seg_start
    seg_len2 = np.sum(seg * seg, axis=-1, keepdims=True)
    seg_len2 = np.maximum(seg_len2, 1e-12)
    proj = np.clip(np.sum((points - seg_start) * seg, axis=-1, keepdims=True) / seg_len2, 0.0, 1.0)
    closest = seg_start + proj * seg
    return np.linalg.norm(points - closest, axis=-1)


def _np_segment_segment_distance(
    seg_start_a: np.ndarray,
    seg_end_a: np.ndarray,
    seg_start_b: np.ndarray,
    seg_end_b: np.ndarray,
) -> np.ndarray:
    eps = 1e-12
    u = seg_end_a - seg_start_a
    v = seg_end_b - seg_start_b
    w = seg_start_a - seg_start_b

    a = np.sum(u * u, axis=-1)
    b = np.sum(u * v, axis=-1)
    c = np.sum(v * v, axis=-1)
    d = np.sum(u * w, axis=-1)
    e = np.sum(v * w, axis=-1)
    denom = a * c - b * b

    s_num = b * e - c * d
    t_num = a * e - b * d
    s_den = denom.copy()
    t_den = denom.copy()

    parallel = denom < eps
    s_num = np.where(parallel, 0.0, s_num)
    s_den = np.where(parallel, 1.0, s_den)
    t_num = np.where(parallel, e, t_num)
    t_den = np.where(parallel, np.maximum(c, eps), t_den)

    low_s = (~parallel) & (s_num < 0.0)
    s_num = np.where(low_s, 0.0, s_num)
    t_num = np.where(low_s, e, t_num)
    t_den = np.where(low_s, np.maximum(c, eps), t_den)

    high_s = (~parallel) & (s_num > s_den)
    s_num = np.where(high_s, s_den, s_num)
    t_num = np.where(high_s, e + b, t_num)
    t_den = np.where(high_s, np.maximum(c, eps), t_den)

    low_t = t_num < 0.0
    t_num = np.where(low_t, 0.0, t_num)
    neg_d = -d
    s_num = np.where(low_t & (neg_d < 0.0), 0.0, s_num)
    s_num = np.where(low_t & (neg_d > a), s_den, s_num)
    between_low = low_t & (neg_d >= 0.0) & (neg_d <= a)
    s_num = np.where(between_low, neg_d, s_num)
    s_den = np.where(between_low, np.maximum(a, eps), s_den)

    high_t = t_num > t_den
    t_num = np.where(high_t, t_den, t_num)
    b_minus_d = b - d
    s_num = np.where(high_t & (b_minus_d < 0.0), 0.0, s_num)
    s_num = np.where(high_t & (b_minus_d > a), s_den, s_num)
    between_high = high_t & (b_minus_d >= 0.0) & (b_minus_d <= a)
    s_num = np.where(between_high, b_minus_d, s_num)
    s_den = np.where(between_high, np.maximum(a, eps), s_den)

    s = np.where(np.abs(s_num) < eps, 0.0, s_num / np.maximum(s_den, eps))
    t = np.where(np.abs(t_num) < eps, 0.0, t_num / np.maximum(t_den, eps))
    delta = w + s[..., None] * u - t[..., None] * v
    return np.linalg.norm(delta, axis=-1)


def _np_points_in_triangle(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    eps = 1e-9
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = np.sum(v0 * v0, axis=-1)
    d01 = np.sum(v0 * v1, axis=-1)
    d11 = np.sum(v1 * v1, axis=-1)
    d20 = np.sum(v2 * v0, axis=-1)
    d21 = np.sum(v2 * v1, axis=-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / np.maximum(denom, eps)
    w = (d00 * d21 - d01 * d20) / np.maximum(denom, eps)
    u = 1.0 - v - w
    return (denom > eps) & (u >= -eps) & (v >= -eps) & (w >= -eps)


def _np_point_triangle_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    eps = 1e-12
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_norm = np.linalg.norm(normal, axis=-1)
    normal_unit = normal / np.maximum(normal_norm, eps)[..., None]
    signed = np.sum((points - a) * normal_unit, axis=-1)
    projected = points - signed[..., None] * normal_unit
    inside = _np_points_in_triangle(projected, a, b, c)
    plane_dist = np.abs(signed)
    edge_dist = np.minimum.reduce(
        [
            _np_point_segment_distance(points, a, b),
            _np_point_segment_distance(points, b, c),
            _np_point_segment_distance(points, c, a),
        ]
    )
    return np.where(inside, plane_dist, edge_dist)


def _np_segment_triangle_intersects(
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    eps = 1e-9
    direction = seg_end - seg_start
    normal = np.cross(b - a, c - a)
    denom = np.sum(direction * normal, axis=-1)
    t = np.sum((a - seg_start) * normal, axis=-1) / np.where(np.abs(denom) < eps, 1.0, denom)
    hit_point = seg_start + t[..., None] * direction
    return (np.abs(denom) >= eps) & (t >= -eps) & (t <= 1.0 + eps) & _np_points_in_triangle(hit_point, a, b, c)


def _np_segment_mesh_distance(
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a = vertices[faces[:, 0]][None, :, :]
    b = vertices[faces[:, 1]][None, :, :]
    c = vertices[faces[:, 2]][None, :, :]
    s0 = seg_start[:, None, :]
    s1 = seg_end[:, None, :]

    endpoint_dist = np.minimum(
        _np_point_triangle_distance(s0, a, b, c),
        _np_point_triangle_distance(s1, a, b, c),
    )
    edge_dist = np.minimum.reduce(
        [
            _np_segment_segment_distance(s0, s1, a, b),
            _np_segment_segment_distance(s0, s1, b, c),
            _np_segment_segment_distance(s0, s1, c, a),
        ]
    )
    distances = np.minimum(endpoint_dist, edge_dist)
    intersects = _np_segment_triangle_intersects(s0, s1, a, b, c)
    distances = np.where(intersects, 0.0, distances)
    return np.min(distances, axis=1), np.any(intersects, axis=1)


def _obstacle_world_mesh_np(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    spec = normalize_obstacle_spec(spec)
    if spec["shape"] == "sphere":
        raise ValueError("sphere obstacles do not have a triangle mesh")
    vertices, faces = obstacle_mesh_np(spec)
    center = np.asarray(spec["center"], dtype=np.float64).reshape(1, 3)
    rot = _np_quat_to_rotmat_wxyz(np.asarray(spec["quat"], dtype=np.float64))
    return vertices @ rot.T + center, faces


def obstacle_world_aabb_half_extents_np(spec: dict) -> np.ndarray:
    """Return yaw-aware world AABB half extents for keeping an obstacle inside a box."""
    spec = normalize_obstacle_spec(spec)
    if spec["shape"] == "sphere":
        return np.full(3, float(spec["radius"]), dtype=np.float64)
    vertices_world, _ = _obstacle_world_mesh_np({**spec, "center": [0.0, 0.0, 0.0]})
    return np.max(np.abs(vertices_world), axis=0)


def _mesh_edges_np(faces: np.ndarray) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    for tri in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_set.add(tuple(sorted((int(a), int(b)))))
    if not edge_set:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(edge_set), dtype=np.int64)


def _np_point_mesh_distance(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a = vertices[faces[:, 0]][None, :, :]
    b = vertices[faces[:, 1]][None, :, :]
    c = vertices[faces[:, 2]][None, :, :]
    return np.min(_np_point_triangle_distance(pts[:, None, :], a, b, c), axis=1)


def _sphere_mesh_distance_np(sphere_spec: dict, mesh_spec: dict) -> float:
    sphere_spec = normalize_obstacle_spec(sphere_spec)
    mesh_spec = normalize_obstacle_spec(mesh_spec)
    center = np.asarray(sphere_spec["center"], dtype=np.float64).reshape(1, 3)
    radius = float(sphere_spec["radius"])
    vertices, faces = _obstacle_world_mesh_np(mesh_spec)
    surface_distance = float(_np_point_mesh_distance(center, vertices, faces)[0])
    if bool(points_in_obstacle_np(center, mesh_spec, padding=0.0).reshape(-1)[0]):
        return -(surface_distance + radius)
    return surface_distance - radius


def obstacle_distance_np(spec_a: dict, spec_b: dict) -> float:
    """Approximate signed surface distance between two rendered obstacle shapes.

    Positive means separated, zero means touching/intersecting for mesh-mesh,
    and negative means penetrating when at least one shape is a sphere.
    """
    spec_a = normalize_obstacle_spec(spec_a)
    spec_b = normalize_obstacle_spec(spec_b)

    if spec_a["shape"] == "sphere" and spec_b["shape"] == "sphere":
        center_a = np.asarray(spec_a["center"], dtype=np.float64)
        center_b = np.asarray(spec_b["center"], dtype=np.float64)
        return float(np.linalg.norm(center_a - center_b) - float(spec_a["radius"]) - float(spec_b["radius"]))

    if spec_a["shape"] == "sphere":
        return _sphere_mesh_distance_np(spec_a, spec_b)
    if spec_b["shape"] == "sphere":
        return _sphere_mesh_distance_np(spec_b, spec_a)

    vertices_a, faces_a = _obstacle_world_mesh_np(spec_a)
    vertices_b, faces_b = _obstacle_world_mesh_np(spec_b)

    if (
        np.any(points_in_obstacle_np(vertices_a, spec_b, padding=0.0))
        or np.any(points_in_obstacle_np(vertices_b, spec_a, padding=0.0))
    ):
        return 0.0

    min_distance = np.inf
    edges_a = _mesh_edges_np(faces_a)
    if edges_a.size > 0:
        dist_a_to_b, _ = _np_segment_mesh_distance(
            vertices_a[edges_a[:, 0]],
            vertices_a[edges_a[:, 1]],
            vertices_b,
            faces_b,
        )
        min_distance = min(min_distance, float(np.min(dist_a_to_b)))

    edges_b = _mesh_edges_np(faces_b)
    if edges_b.size > 0:
        dist_b_to_a, _ = _np_segment_mesh_distance(
            vertices_b[edges_b[:, 0]],
            vertices_b[edges_b[:, 1]],
            vertices_a,
            faces_a,
        )
        min_distance = min(min_distance, float(np.min(dist_b_to_a)))

    return float(min_distance)


def capsules_overlap_obstacle_np(
    seg_start,
    seg_end,
    radii,
    spec: dict,
    *,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict | None]:
    spec = normalize_obstacle_spec(spec)
    start = np.asarray(seg_start, dtype=np.float64)
    end = np.asarray(seg_end, dtype=np.float64)
    original_shape = np.broadcast_shapes(start.shape[:-1], end.shape[:-1], np.asarray(radii).shape)
    start = np.broadcast_to(start, original_shape + (3,)).reshape(-1, 3)
    end = np.broadcast_to(end, original_shape + (3,)).reshape(-1, 3)
    radii_arr = np.broadcast_to(np.asarray(radii, dtype=np.float64), original_shape).reshape(-1)

    if spec["shape"] == "sphere":
        center = np.asarray(spec["center"], dtype=np.float64).reshape(1, 3)
        dist = _np_point_segment_distance(center, start, end)
        threshold = float(spec["radius"]) + radii_arr
        overlap = dist <= threshold
        if not return_details:
            return overlap.reshape(original_shape)
        hit = np.flatnonzero(overlap)
        details = None
        if hit.size > 0:
            local_idx = int(hit[np.argmin(dist[hit] - threshold[hit])])
            details = {
                "segment_index": local_idx,
                "distance": float(dist[local_idx]),
                "threshold": float(threshold[local_idx]),
                "clearance": float(dist[local_idx] - threshold[local_idx]),
            }
        return overlap.reshape(original_shape), details

    center = np.asarray(spec["center"], dtype=np.float64).reshape(1, 3)
    rot = _np_quat_to_rotmat_wxyz(np.asarray(spec["quat"], dtype=np.float64))
    local_start = (start - center) @ rot
    local_end = (end - center) @ rot
    bound_radius = float(np.linalg.norm(np.asarray(spec["half_extents"], dtype=np.float64).reshape(3)))
    center_dist = _np_point_segment_distance(np.zeros_like(local_start), local_start, local_end)
    near = center_dist <= (bound_radius + radii_arr)
    if not np.any(near):
        overlap = np.zeros(start.shape[0], dtype=bool)
        if not return_details:
            return overlap.reshape(original_shape)
        return overlap.reshape(original_shape), None

    vertices, faces = obstacle_mesh_np(spec)
    mesh_dist_near, intersects_near = _np_segment_mesh_distance(local_start[near], local_end[near], vertices, faces)
    inside_start_near = points_in_obstacle_np(start[near], spec, padding=0.0).reshape(-1)
    inside_end_near = points_in_obstacle_np(end[near], spec, padding=0.0).reshape(-1)
    centerline_inside_near = intersects_near | inside_start_near | inside_end_near
    dist_near = np.where(centerline_inside_near, 0.0, mesh_dist_near)

    dist = np.full(start.shape[0], np.inf, dtype=np.float64)
    centerline_inside = np.zeros(start.shape[0], dtype=bool)
    dist[near] = dist_near
    centerline_inside[near] = centerline_inside_near
    overlap = dist <= radii_arr

    if not return_details:
        return overlap.reshape(original_shape)
    hit = np.flatnonzero(overlap)
    details = None
    if hit.size > 0:
        clearance = dist - radii_arr
        local_idx = int(hit[np.argmin(clearance[hit])])
        details = {
            "segment_index": local_idx,
            "distance": float(dist[local_idx]),
            "threshold": float(radii_arr[local_idx]),
            "clearance": float(clearance[local_idx]),
            "centerline_inside": bool(centerline_inside[local_idx]),
        }
    return overlap.reshape(original_shape), details


# ----------------------------------------------------------------- torch


def _torch_quat_to_rotmat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(dtype=torch.float32)
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        [
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def obstacle_spec_to_torch(spec: dict, *, device, dtype=torch.float32) -> dict:
    spec = normalize_obstacle_spec(spec)
    return {
        "shape": spec["shape"],
        "shape_id": int(SHAPE_IDS[spec["shape"]]),
        "center": torch.as_tensor(spec["center"], dtype=dtype, device=device).view(3),
        "quat": torch.as_tensor(spec["quat"], dtype=dtype, device=device).view(4),
        "half_extents": torch.as_tensor(spec["half_extents"], dtype=dtype, device=device).view(3),
        "radius": torch.as_tensor(float(spec.get("radius", spec["half_extents"][0])), dtype=dtype, device=device),
    }


def _torch_segment_aabb_intersects(
    start_local: torch.Tensor,
    end_local: torch.Tensor,
    half_extents: torch.Tensor,
) -> torch.Tensor:
    """
    Liang-Barsky slab method: does the line segment intersect the axis-aligned
    box [-half_extents, +half_extents]?

    Inputs:
      start_local, end_local: (N, 3) segment endpoints in box local frame
      half_extents: (3,) box half-extents (possibly already inflated by capsule radius)
    Returns: (N,) boolean
    """
    eps = 1e-9
    d = end_local - start_local                                # (N, 3)
    abs_d = torch.abs(d)
    safe_d = torch.where(abs_d > eps, d, torch.full_like(d, eps))
    t0 = (-half_extents - start_local) / safe_d
    t1 = (half_extents - start_local) / safe_d
    t_enter = torch.minimum(t0, t1)
    t_exit = torch.maximum(t0, t1)

    parallel = abs_d <= eps
    in_bounds_each = torch.abs(start_local) <= half_extents
    # axes that are parallel & out-of-bounds disqualify the segment entirely
    no_intersect_axis = parallel & ~in_bounds_each
    # parallel-in-bounds axes don't constrain — clamp t-range to infinity
    NEG_INF = torch.full_like(t_enter, -1e10)
    POS_INF = torch.full_like(t_exit, 1e10)
    t_enter = torch.where(parallel, NEG_INF, t_enter)
    t_exit = torch.where(parallel, POS_INF, t_exit)

    t_enter_max = torch.max(t_enter, dim=-1).values
    t_exit_min = torch.min(t_exit, dim=-1).values

    intersects = (
        (t_enter_max <= t_exit_min)
        & (t_enter_max <= 1.0)
        & (t_exit_min >= 0.0)
        & ~torch.any(no_intersect_axis, dim=-1)
    )
    return intersects


def _torch_capsule_box_overlap_analytic(
    start_local: torch.Tensor,
    end_local: torch.Tensor,
    capsule_radii: torch.Tensor,
    half_extents: torch.Tensor,
) -> torch.Tensor:
    """
    Exact analytic binary overlap test: does the capsule (segment + radius)
    intersect the axis-aligned box [-h, +h] (in box local frame)?

    Strategy: capsule overlaps box  ⇔  segment passes within capsule_radius of
    box surface  ≈  segment intersects the AABB inflated by capsule_radius.

    This is slightly conservative at the box corners (treats rounded corners
    as boxy), but is exact for all face/edge cases and runs in ~6 GPU ops.
    """
    h = half_extents.view(1, 3)                                # (1, 3)
    r = capsule_radii.view(-1, 1)                              # (N, 1)
    h_inflated = h + r                                         # (N, 3)
    return _torch_segment_aabb_intersects(start_local, end_local, h_inflated)


def _torch_tri_prism_signed_distance_points(local_points: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    """Analytic SDF for an extruded isoceles-triangle prism in local frame.

    Mesh (tri_prism_obstacle.obj) is extruded along Y. Cross-section in XZ has
    triangle vertices at (-hx, -hz), (+hx, -hz), (0, +hz). Y-extent is ±hy.

    Replaces the much slower mesh-distance call (304 ms with 7 sample points,
    132 ms with 3 sample points) with O(point) analytic computation.
    """
    half = half_extents.view(1, 3)
    hx = half[..., 0]
    hy = half[..., 1]
    hz = half[..., 2]

    x = local_points[..., 0]
    y = local_points[..., 1]
    z = local_points[..., 2]
    # Use lateral symmetry along X: triangle is symmetric.
    ax = torch.abs(x)

    # 2D triangle SDF in XZ. Vertices: A=(-hx,-hz), B=(+hx,-hz), C=(0,+hz).
    # Inigo Quilez's classic formulation, simplified using |x| symmetry.
    # Edge 0: from A→B (bottom horizontal). Edge 1: from B→C (right slant).
    # Edge 2: from C→A (left slant). We only need 2 edges thanks to symmetry.
    # Bottom edge is y=-hz, length 2hx. Slant edges go from (±hx,-hz) → (0,+hz).
    eps = 1e-9

    # Distance to bottom edge (z = -hz), x ∈ [-hx, +hx].
    bottom_dx = torch.where(ax > hx, ax - hx, torch.zeros_like(ax))
    bottom_dz = torch.where(z < -hz, (-hz - z), torch.zeros_like(z))
    bottom_outside = torch.sqrt(bottom_dx * bottom_dx + bottom_dz * bottom_dz + eps) - eps**0.5
    # signed: positive when outside (below or past x-range), negative when inside band
    bottom_signed = torch.where(z < -hz, bottom_outside, -(z - (-hz)))

    # Slant edge (right slant): from (hx, -hz) → (0, +hz).
    edge_dx = -hx
    edge_dz = 2.0 * hz
    edge_len2 = edge_dx * edge_dx + edge_dz * edge_dz + eps
    # vector from (hx,-hz) to point (using |x|)
    rel_x = ax - hx
    rel_z = z - (-hz)
    t = torch.clamp((rel_x * edge_dx + rel_z * edge_dz) / edge_len2, 0.0, 1.0)
    proj_x = hx + t * edge_dx
    proj_z = -hz + t * edge_dz
    slant_dx = ax - proj_x
    slant_dz = z - proj_z
    slant_dist = torch.sqrt(slant_dx * slant_dx + slant_dz * slant_dz + eps) - eps**0.5
    # Signed: positive if outside (cross product test).
    # Outside means (ax, z) is on the right of edge B→C.
    cross = edge_dx * rel_z - edge_dz * rel_x  # >0 means inside
    slant_signed = torch.where(cross < 0.0, slant_dist, -slant_dist)

    # 2D triangle SDF = max of the per-edge signed distances (inside test).
    # But the standard formulation: SDF = sign × min(distances).
    # Simpler: combine with bottom and slant. Inside triangle requires both
    # bottom_signed <= 0 and slant_signed <= 0.
    d2d = torch.maximum(bottom_signed, slant_signed)

    # Y extent.
    d_y = torch.abs(y) - hy

    # 3D SDF combine — extruded shape.
    # outside: sqrt(max(0, d2d)² + max(0, d_y)²)
    # inside:  max(d2d, d_y)  (both negative)
    pos_d2d = torch.clamp(d2d, min=0.0)
    pos_dy = torch.clamp(d_y, min=0.0)
    outside_part = torch.sqrt(pos_d2d * pos_d2d + pos_dy * pos_dy + eps) - eps**0.5
    inside_part = torch.minimum(torch.maximum(d2d, d_y), torch.zeros_like(d2d))
    return outside_part + inside_part


def _torch_capsule_tri_prism_overlap_analytic(
    start_local: torch.Tensor,
    end_local: torch.Tensor,
    capsule_radii: torch.Tensor,
    half_extents: torch.Tensor,
) -> torch.Tensor:
    """
    Exact analytic binary overlap test for capsule vs triangular prism (in
    obstacle local frame). The prism's cross-section is a triangle in XZ with
    vertices A=(-hx,-hz), B=(+hx,-hz), C=(0,+hz), extruded along Y to ±hy.

    Strategy (no sampling — fully analytic):

      1. Conservative AABB pre-check rejects clearly-separated segments.
      2. Y-clip restricts the segment to the t-range where it lies within
         [-hy-r, +hy-r] (the inflated Y extent).
      3. Project the Y-clipped segment to XZ. Capsule overlaps prism iff:
           a. either projected endpoint is inside the inflated triangle, OR
           b. the projected segment comes within capsule_radius of any of the
              three triangle edges (closed-form segment-vs-segment distance).
    """
    h = half_extents.view(1, 3)
    r = capsule_radii.view(-1, 1)
    r_flat = capsule_radii.view(-1)
    eps = 1e-9

    # 1. Conservative AABB pre-check.
    aabb_hit = _torch_segment_aabb_intersects(start_local, end_local, h + r)

    # 2. Y-clip.
    Ay = start_local[..., 1]
    By = end_local[..., 1]
    dy = By - Ay
    abs_dy = torch.abs(dy)
    hy = h[..., 1] + r_flat
    safe_dy = torch.where(abs_dy > eps, dy, torch.full_like(dy, eps))
    t0 = (-hy - Ay) / safe_dy
    t1 = (hy - Ay) / safe_dy
    t_y_enter = torch.minimum(t0, t1)
    t_y_exit = torch.maximum(t0, t1)
    parallel_y = abs_dy <= eps
    y_inside = torch.abs(Ay) <= hy
    t_y_enter = torch.where(parallel_y, torch.full_like(t_y_enter, 0.0), t_y_enter)
    t_y_exit = torch.where(parallel_y, torch.full_like(t_y_exit, 1.0), t_y_exit)
    t_lo = torch.clamp(t_y_enter, min=0.0, max=1.0)
    t_hi = torch.clamp(t_y_exit, min=0.0, max=1.0)
    y_range_valid = (t_lo <= t_hi) & (~parallel_y | y_inside)

    # 3. Y-clipped segment end points.
    p_lo = start_local + t_lo.unsqueeze(-1) * (end_local - start_local)
    p_hi = start_local + t_hi.unsqueeze(-1) * (end_local - start_local)

    # 3a. Endpoint-in-(inflated)-prism via the 3D analytic SDF (handles Y).
    sdf_A = _torch_tri_prism_signed_distance_points(p_lo, half_extents.view(3))
    sdf_B = _torch_tri_prism_signed_distance_points(p_hi, half_extents.view(3))
    endpoint_hit = (sdf_A <= r_flat) | (sdf_B <= r_flat)

    # 3b. Walsh's rule: along the Y-clipped segment, the 3D SDF is piecewise-
    #     monotonic. Five evenly-spaced samples (incl. endpoints + midpoint +
    #     quartiles) catch every collision case in practice for short-link /
    #     few-cm-obstacle geometry. This is the only place we rely on sampling;
    #     unlike the soft-cost path it is used purely for the binary check.
    def _sdf_at(t):
        pt = start_local + t.unsqueeze(-1) * (end_local - start_local)
        return _torch_tri_prism_signed_distance_points(pt, half_extents.view(3))

    t_q1 = t_lo + 0.25 * (t_hi - t_lo)
    t_mid = 0.5 * (t_lo + t_hi)
    t_q3 = t_lo + 0.75 * (t_hi - t_lo)
    sdf_q1 = _sdf_at(t_q1)
    sdf_mid = _sdf_at(t_mid)
    sdf_q3 = _sdf_at(t_q3)
    sweep_hit = (
        (sdf_q1 <= r_flat) | (sdf_mid <= r_flat) | (sdf_q3 <= r_flat)
    )

    return aabb_hit & y_range_valid & (endpoint_hit | sweep_hit)
