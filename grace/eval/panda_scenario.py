"""Random trials of the FR3 evaluation: start/goal configurations and unseen obstacles.

Trial i of a run with seed s uses the seed s + i (if a trial cannot be built,
the next 31 seeds are tried). Start and goal are drawn with
``sample_episode_endpoints`` and must additionally pass conservative checks of
the capsule model (joint-limit margin, inflated self/floor clearance). Six
obstacles (2 spheres, 2 boxes, 2 triangular prisms) are then placed with a
separate random generator (seed + 1009) in the workspace, away from the arm at
start and goal and from each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from grace.envs.panda.capsules import (
    PandaCapsuleFK,
    capsule_radii,
    gripper_capsule_indices,
    joint_limits,
    self_collision_pairs,
)
from grace.envs.panda.kinematics import PandaPinocchio
from grace.envs.panda.obstacles import (
    box_obstacle,
    capsules_overlap_obstacle_np,
    clone_obstacle_specs,
    normalize_shape_name,
    obstacle_distance_np,
    obstacle_world_aabb_half_extents_np,
    sphere_obstacle,
    tri_prism_obstacle,
)
from grace.envs.panda.sampling import sample_episode_endpoints
from grace.envs.panda.sim import PandaDemoSim


@dataclass(frozen=True)
class ScenarioConfig:
    """Paper settings of the FR3 evaluation scenarios."""

    # conservative start/goal checks
    collision_margin: float = 1.0
    floor_height: float = 0.01
    joint_limit_extra_margin: float = 0.02
    collision_radius_padding: float = 0.005
    floor_extra_clearance: float = 0.005
    # obstacles
    obstacle_shapes: tuple = ("sphere", "sphere", "box", "box", "tri_prism", "tri_prism")
    sphere_radius: float = 0.1
    box_size_radius: float = 0.07
    workspace_padding_xyz: tuple = (0.0, 0.0, 0.1)
    min_distance_to_start_goal: float = 0.10
    min_distance_between_obstacles: float = 0.10
    max_sampling_tries: int = 4000
    seed_offsets: int = 32


@dataclass
class Trial:
    seed: int
    q_start: np.ndarray
    start_pos: np.ndarray
    q_goal: np.ndarray
    goal_pos: np.ndarray
    goal_quat_xyzw: np.ndarray
    obstacles: list


# ------------------------------------------------------------------ capsule checks (numpy)
@lru_cache(maxsize=1)
def _capsule_fk_cpu():
    return PandaCapsuleFK(device="cpu")


def _capsule_segments_for_q(q_arm):
    q_arm = np.asarray(q_arm, dtype=float).reshape(-1)
    seg_start, seg_end = _capsule_fk_cpu().forward_collision_segments(torch.as_tensor(q_arm, dtype=torch.float32).view(1, -1))
    return seg_start[0].detach().cpu().numpy()[:, :3], seg_end[0].detach().cpu().numpy()[:, :3]


def _capsule_radii_padded(link_padding, gripper_padding):
    radii = np.asarray(capsule_radii(), dtype=float).reshape(-1)
    radii = radii + max(0.0, float(link_padding))
    if gripper_padding > 0.0 and radii.size > 0:
        idx = np.asarray(gripper_capsule_indices(), dtype=int).reshape(-1)
        idx = idx[(idx >= 0) & (idx < radii.size)]
        if idx.size > 0:
            radii[idx] = radii[idx] + max(0.0, float(gripper_padding))
    return radii


def _point_segment_distance(point, seg_start, seg_end):
    seg = seg_end - seg_start
    seg_len2 = np.maximum(np.sum(seg * seg, axis=-1, keepdims=True), 1e-8)
    rel = point.reshape(1, 3) - seg_start
    proj = np.clip(np.sum(rel * seg, axis=-1, keepdims=True) / seg_len2, 0.0, 1.0)
    closest = seg_start + proj * seg
    return np.linalg.norm(closest - point.reshape(1, 3), axis=-1)


def _segment_segment_distance_np(seg_start_a, seg_end_a, seg_start_b, seg_end_b):
    eps = 1e-8
    u = seg_end_a - seg_start_a
    v = seg_end_b - seg_start_b
    w = seg_start_a - seg_start_b
    a = np.sum(u * u, axis=-1)
    b = np.sum(u * v, axis=-1)
    c = np.sum(v * v, axis=-1)
    d = np.sum(u * w, axis=-1)
    e = np.sum(v * w, axis=-1)
    D = a * c - b * b
    sN = b * e - c * d
    tN = a * e - b * d
    sD = D.copy()
    tD = D.copy()
    parallel = D < eps
    sN = np.where(parallel, 0.0, sN)
    sD = np.where(parallel, 1.0, sD)
    tN = np.where(parallel, e, tN)
    tD = np.where(parallel, np.maximum(c, eps), tD)
    clamp_s_low = (~parallel) & (sN < 0.0)
    sN = np.where(clamp_s_low, 0.0, sN)
    tN = np.where(clamp_s_low, e, tN)
    tD = np.where(clamp_s_low, np.maximum(c, eps), tD)
    clamp_s_high = (~parallel) & (sN > sD)
    sN = np.where(clamp_s_high, sD, sN)
    tN = np.where(clamp_s_high, e + b, tN)
    tD = np.where(clamp_s_high, np.maximum(c, eps), tD)
    clamp_t_low = tN < 0.0
    tN = np.where(clamp_t_low, 0.0, tN)
    neg_d = -d
    sN = np.where(clamp_t_low & (neg_d < 0.0), 0.0, sN)
    sN = np.where(clamp_t_low & (neg_d > a), sD, sN)
    between_low = clamp_t_low & (neg_d >= 0.0) & (neg_d <= a)
    sN = np.where(between_low, neg_d, sN)
    sD = np.where(between_low, np.maximum(a, eps), sD)
    clamp_t_high = tN > tD
    tN = np.where(clamp_t_high, tD, tN)
    b_minus_d = b - d
    sN = np.where(clamp_t_high & (b_minus_d < 0.0), 0.0, sN)
    sN = np.where(clamp_t_high & (b_minus_d > a), sD, sN)
    between_high = clamp_t_high & (b_minus_d >= 0.0) & (b_minus_d <= a)
    sN = np.where(between_high, b_minus_d, sN)
    sD = np.where(between_high, np.maximum(a, eps), sD)
    sc = np.where(np.abs(sN) < eps, 0.0, sN / np.maximum(sD, eps))
    tc = np.where(np.abs(tN) < eps, 0.0, tN / np.maximum(tD, eps))
    delta = w + sc[..., None] * u - tc[..., None] * v
    return np.linalg.norm(delta, axis=-1)


def joint_limit_violation(q_arm, tolerance):
    """Largest violation of the joint limits shrunk by ``tolerance`` (0 if none)."""
    lower, upper = (np.asarray(v, dtype=float) for v in joint_limits())
    q_arm = np.asarray(q_arm, dtype=float).reshape(-1)[: lower.shape[0]]
    tol = max(0.0, float(tolerance))
    lower_violation = np.maximum((lower + tol) - q_arm, 0.0)
    upper_violation = np.maximum(q_arm - (upper - tol), 0.0)
    return float(np.max(lower_violation + upper_violation))


def _joint_limits_ok(q_arm, margin):
    lower, upper = (np.asarray(v, dtype=float) for v in joint_limits())
    q_arm = np.asarray(q_arm, dtype=float).reshape(-1)[: lower.shape[0]]
    return bool(np.all(q_arm >= (lower + margin)) and np.all(q_arm <= (upper - margin)))


def _capsule_self_floor_collision(q_arm, collision_margin, radius_padding, floor_height, floor_clearance):
    seg_start, seg_end = _capsule_segments_for_q(q_arm)
    seg_radii = _capsule_radii_padded(radius_padding, radius_padding)[: seg_start.shape[0]]

    pair_idx = np.asarray(self_collision_pairs(), dtype=int).reshape(-1, 2)
    dist = _segment_segment_distance_np(
        seg_start[pair_idx[:, 0]], seg_end[pair_idx[:, 0]], seg_start[pair_idx[:, 1]], seg_end[pair_idx[:, 1]]
    )
    pair_radii = (seg_radii[pair_idx[:, 0]] + seg_radii[pair_idx[:, 1]]) * float(collision_margin)
    self_collision = bool(np.any(dist < pair_radii))

    seg_min_z = np.minimum(seg_start[1:, 2], seg_end[1:, 2])
    threshold = float(floor_height) + float(collision_margin) * seg_radii[1:] + max(0.0, float(floor_clearance))
    floor_collision = bool(np.any(seg_min_z < threshold))
    return self_collision, floor_collision


def obstacle_overlaps_arm(q_arm, spec, collision_margin, extra_radius_padding):
    """Capsule model of the arm at ``q_arm`` (radii + padding) overlaps the obstacle."""
    seg_start, seg_end = _capsule_segments_for_q(q_arm)
    seg_radii = _capsule_radii_padded(0.0, 0.0)[: seg_start.shape[0]]
    seg_radii = seg_radii + max(0.0, float(extra_radius_padding))
    spec = clone_obstacle_specs([spec])[0]
    if spec["shape"] == "sphere":
        center = np.asarray(spec["center"], dtype=float).reshape(3)
        dist = _point_segment_distance(center, seg_start, seg_end)
        return bool(np.any(dist < float(spec["radius"]) * float(collision_margin) + seg_radii))
    overlap = capsules_overlap_obstacle_np(seg_start, seg_end, seg_radii * float(collision_margin), spec)
    return bool(np.any(overlap))


# ------------------------------------------------------------------ trial construction
def sample_start_goal(config, scenario, seed, kin):
    rng = np.random.default_rng(seed)
    checker = PandaDemoSim(config)
    last_exc = None
    for _ in range(max(1, int(config.max_sampling_tries))):
        try:
            start_pos, _, q_start, goal_pos, goal_rot, q_goal = sample_episode_endpoints(rng, kin, config, checker)
        except Exception as exc:
            last_exc = exc
            continue

        def conservative_ok(q):
            if not _joint_limits_ok(q, max(0.0, float(config.joint_margin)) + max(0.0, scenario.joint_limit_extra_margin)):
                return False
            self_c, floor_c = _capsule_self_floor_collision(
                q,
                scenario.collision_margin,
                scenario.collision_radius_padding,
                scenario.floor_height,
                scenario.floor_extra_clearance,
            )
            return (not self_c) and (not floor_c)

        if conservative_ok(q_start) and conservative_ok(q_goal):
            return (
                np.asarray(q_start, dtype=float),
                np.asarray(start_pos, dtype=float),
                np.asarray(q_goal, dtype=float),
                np.asarray(goal_pos, dtype=float),
                Rotation.from_matrix(np.asarray(goal_rot, dtype=float)).as_quat(),
            )
    raise RuntimeError("Failed to sample a start/goal pair that passes the conservative checks.") from last_exc


def _obstacle_spec(shape, center, radius, yaw, idx):
    shape = normalize_shape_name(shape)
    if shape == "sphere":
        return sphere_obstacle(center[0], center[1], center[2], radius, name=f"obstacle_{idx:02d}_sphere")
    if shape == "box":
        half_extents = np.asarray([0.45 * radius, 2.2 * radius, 1.6 * radius], dtype=np.float32)
        half_extents = np.maximum(half_extents, np.asarray([0.022, 0.10, 0.075], dtype=np.float32))
        return box_obstacle(center, half_extents, yaw=yaw, name=f"obstacle_{idx:02d}_box")
    return tri_prism_obstacle(center, yaw=yaw, name=f"obstacle_{idx:02d}_tri_prism")


def sample_obstacles(config, scenario, q_start, q_goal, seed):
    shapes = [normalize_shape_name(s) for s in scenario.obstacle_shapes]
    count = len(shapes)
    padding = np.asarray(scenario.workspace_padding_xyz, dtype=float)
    low = np.asarray(config.workspace_low, dtype=float) - padding
    high = np.asarray(config.workspace_high, dtype=float) + padding
    rng = np.random.default_rng(int(seed) + 1009)
    obstacles = []
    for _ in range(max(count, int(scenario.max_sampling_tries))):
        if len(obstacles) >= count:
            break
        shape = shapes[len(obstacles)]
        r = scenario.box_size_radius if shape == "box" else scenario.sphere_radius
        radius = float(rng.uniform(r, r))  # (a draw is consumed although the range is a single value)
        yaw = float(rng.uniform(-np.pi, np.pi))
        provisional = _obstacle_spec(shape, np.zeros(3, dtype=float), radius, yaw, len(obstacles))
        half = obstacle_world_aabb_half_extents_np(provisional).astype(float)
        sample_low = low + half
        sample_high = high - half
        sample_low[2] = max(sample_low[2], half[2])
        if np.any(sample_low >= sample_high):
            raise ValueError("random obstacle sampling box is invalid")
        center = rng.uniform(sample_low, sample_high)
        candidate = _obstacle_spec(shape, center, radius, yaw, len(obstacles))
        if obstacle_overlaps_arm(q_start, candidate, scenario.collision_margin, scenario.min_distance_to_start_goal):
            continue
        if obstacle_overlaps_arm(q_goal, candidate, scenario.collision_margin, scenario.min_distance_to_start_goal):
            continue
        if any(obstacle_distance_np(candidate, obs) < scenario.min_distance_between_obstacles for obs in obstacles):
            continue
        obstacles.append(candidate)
    if len(obstacles) != count:
        raise RuntimeError("Failed to sample the requested number of obstacles.")
    return obstacles


def make_trial(config, scenario, base_seed, trial_index, kin=None):
    kin = kin or PandaPinocchio(ee_frame=config.ee_frame)
    last_exc = None
    for offset in range(scenario.seed_offsets):
        seed = int(base_seed) + int(trial_index) + offset
        try:
            q_start, start_pos, q_goal, goal_pos, goal_quat = sample_start_goal(config, scenario, seed, kin)
            obstacles = sample_obstacles(config, scenario, q_start, q_goal, seed)
            return Trial(seed, q_start, start_pos, q_goal, goal_pos, goal_quat, obstacles)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError("Failed to initialize trial {}".format(trial_index)) from last_exc
