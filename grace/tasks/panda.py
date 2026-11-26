"""Rollout model and safety cost of the FR3 task for GRACE (paper Sec. IV-A).

Rollout: the candidate delta_q action sequences are executed through a model of
the runtime controller. Each action is clipped and integrated into a latched
joint target (q_des_{t+1} = q_des_t + a_t, starting from the measured q at
planning time), the desired velocity is (q_des_{t+1} - q_des_t) / dt, and the
joint-space PD law qacc = kp (q_des - q) + kd (qd_des - qd) is integrated with
20 explicit substeps per control period (velocity-clipped). The rollout covers
all H = 32 predicted steps.

Cost (replaces w_obs J_obs in eq. (31)):

    J_safe = w_obs J_obs + w_self J_self + w_floor J_floor + w_jl J_jl

Each term counts, over the rolled-out states x_1..x_H, binary violations of the
exact geometry (no soft penalty in 3D): J_obs counts per state the obstacles
that overlap any arm capsule; J_self the states with a capsule-capsule overlap
between bodies more than two hops apart; J_floor the states with a capsule
(except the base) below the floor height; J_jl the states with a joint outside
its limits shrunk by ``joint_limit_margin``. The goal is not part of the cost.
"""

from __future__ import annotations

import numpy as np
import torch

from grace.envs.panda.capsules import PandaCapsuleFK, capsule_radii, joint_limits, self_collision_pairs
from grace.envs.panda.obstacles import (
    _torch_capsule_box_overlap_analytic,
    _torch_capsule_tri_prism_overlap_analytic,
    _torch_quat_to_rotmat_wxyz,
    clone_obstacle_specs,
    obstacle_spec_to_torch,
)


def _segment_segment_distance(seg_start_a, seg_end_a, seg_start_b, seg_end_b):
    """Batched shortest distance between 3D segments ([..., 3] -> [...])."""
    eps = 1e-8
    u = seg_end_a - seg_start_a
    v = seg_end_b - seg_start_b
    w = seg_start_a - seg_start_b

    a = torch.sum(u * u, dim=-1)
    b = torch.sum(u * v, dim=-1)
    c = torch.sum(v * v, dim=-1)
    d = torch.sum(u * w, dim=-1)
    e = torch.sum(v * w, dim=-1)
    D = a * c - b * b

    sN = b * e - c * d
    tN = a * e - b * d
    sD = D.clone()
    tD = D.clone()

    parallel = D < eps
    sN = torch.where(parallel, torch.zeros_like(sN), sN)
    sD = torch.where(parallel, torch.ones_like(sD), sD)
    tN = torch.where(parallel, e, tN)
    tD = torch.where(parallel, c.clamp_min(eps), tD)

    clamp_s_low = (~parallel) & (sN < 0.0)
    sN = torch.where(clamp_s_low, torch.zeros_like(sN), sN)
    tN = torch.where(clamp_s_low, e, tN)
    tD = torch.where(clamp_s_low, c.clamp_min(eps), tD)

    clamp_s_high = (~parallel) & (sN > sD)
    sN = torch.where(clamp_s_high, sD, sN)
    tN = torch.where(clamp_s_high, e + b, tN)
    tD = torch.where(clamp_s_high, c.clamp_min(eps), tD)

    clamp_t_low = tN < 0.0
    tN = torch.where(clamp_t_low, torch.zeros_like(tN), tN)
    neg_d = -d
    sN = torch.where(clamp_t_low & (neg_d < 0.0), torch.zeros_like(sN), sN)
    sN = torch.where(clamp_t_low & (neg_d > a), sD, sN)
    between_low = clamp_t_low & (neg_d >= 0.0) & (neg_d <= a)
    sN = torch.where(between_low, neg_d, sN)
    sD = torch.where(between_low, a.clamp_min(eps), sD)

    clamp_t_high = tN > tD
    tN = torch.where(clamp_t_high, tD, tN)
    b_minus_d = b - d
    sN = torch.where(clamp_t_high & (b_minus_d < 0.0), torch.zeros_like(sN), sN)
    sN = torch.where(clamp_t_high & (b_minus_d > a), sD, sN)
    between_high = clamp_t_high & (b_minus_d >= 0.0) & (b_minus_d <= a)
    sN = torch.where(between_high, b_minus_d, sN)
    sD = torch.where(between_high, a.clamp_min(eps), sD)

    sc = torch.where(torch.abs(sN) < eps, torch.zeros_like(sN), sN / sD.clamp_min(eps))
    tc = torch.where(torch.abs(tN) < eps, torch.zeros_like(tN), tN / tD.clamp_min(eps))
    delta = w + sc.unsqueeze(-1) * u - tc.unsqueeze(-1) * v
    return torch.norm(delta, dim=-1)


class PandaGuidanceTask:
    """bind(obs) -> rollout_cost for GRACE; see the module docstring."""

    def __init__(
        self,
        config,
        normalizer,
        device,
        *,
        w_obs,
        w_self,
        w_floor,
        w_jl,
        floor_height,
        joint_limit_margin,
        action_clip,
    ):
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.w_obs = float(w_obs)
        self.w_self = float(w_self)
        self.w_floor = float(w_floor)
        self.w_jl = float(w_jl)
        self.floor_height = float(floor_height)

        def f32(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device).view(-1)

        # controller model (same gains / limits as the simulation)
        self.control_dt = float(config.control_dt)
        self.substeps = max(1, int(config.substeps))
        self.max_joint_velocity = f32(config.max_joint_velocity)
        self.torque_kp = f32(config.torque_kp)
        self.torque_kd = f32(config.torque_kd)
        self.action_clip = f32(action_clip)

        # capsule model
        self.fk = PandaCapsuleFK(device=self.device, finger_q=float(config.finger_q))
        self.capsule_radii = torch.tensor(capsule_radii(), dtype=torch.float32, device=self.device)
        pair_idx = torch.tensor(self_collision_pairs(), dtype=torch.long, device=self.device)
        self.self_pairs = pair_idx
        self.self_pair_radii = self.capsule_radii[pair_idx[:, 0]] + self.capsule_radii[pair_idx[:, 1]]

        # joint limits shrunk by the margin (at most to the joint's mid-range)
        lower, upper = joint_limits()
        lower = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper = torch.tensor(upper, dtype=torch.float32, device=self.device)
        half_range = 0.5 * (upper - lower)
        margin = torch.clamp(
            torch.full_like(half_range, float(joint_limit_margin)),
            max=(half_range - 1e-6).clamp_min(0.0),
        )
        self.jl_lower = lower + margin
        self.jl_upper = upper - margin

        self.obstacles = []
        self.q_des = None

    # ------------------------------------------------------------ context
    def set_obstacles(self, specs):
        self.obstacles = [obstacle_spec_to_torch(spec, device=self.device) for spec in clone_obstacle_specs(specs)]

    def set_latched_q_des(self, q_des):
        """Joint target the next plan is integrated from (the measured q at planning time)."""
        self.q_des = None if q_des is None else np.asarray(q_des, dtype=np.float32).copy()

    def bind(self, obs):
        q0 = self.normalizer.unnormalize_obs("robot0_joint_pos", obs["robot0_joint_pos"][:, -1, :]).view(-1)
        qd0 = self.normalizer.unnormalize_obs("robot0_joint_vel", obs["robot0_joint_vel"][:, -1, :]).view(-1)
        if self.q_des is None:
            raise RuntimeError("set_latched_q_des() must be called before planning")
        q_des0 = torch.as_tensor(self.q_des, dtype=torch.float32, device=self.device).view(-1)

        def rollout_cost(V):
            return self.safety_cost(self.rollout(q0, qd0, q_des0, V))

        return rollout_cost

    # ------------------------------------------------------------ rollout
    def rollout(self, q0, qd0, q_des0, V):
        """q0, qd0, q_des0: [7]; V: [K, H, 7] delta_q actions -> joint trajectory [K, H, 7]."""
        dt = self.control_dt
        V = V.to(device=self.device, dtype=torch.float32)
        K, H, m = V.shape
        q = q0[:m].view(1, m).repeat(K, 1)
        qd = qd0[:m].view(1, m).repeat(K, 1)
        q_des = q_des0[:m].view(1, m).repeat(K, 1)
        q_des_integrated = q_des.clone()
        prev_desired_q = q_des.clone()

        v_max = self.max_joint_velocity[:m].view(1, m)
        kp = self.torque_kp[:m].view(1, m)
        kd = self.torque_kd[:m].view(1, m)
        integration_dt = dt / float(self.substeps)
        substep_index = torch.arange(1, self.substeps + 1, device=self.device, dtype=torch.float32).view(1, self.substeps, 1)
        clip = self.action_clip[:m].view(1, m)

        q_trajs = []
        for t in range(H):
            a_t = V[:, t, :m]
            a_t = torch.maximum(torch.minimum(a_t, clip), -clip)
            q_des_integrated = q_des_integrated + a_t
            q_des = q_des_integrated
            qd_des = torch.clamp((q_des - prev_desired_q) / dt, -v_max, v_max)
            prev_desired_q = q_des
            qacc_des = kp * (q_des - q) + kd * (qd_des - qd)
            qd_substeps = qd.unsqueeze(1) + integration_dt * substep_index * qacc_des.unsqueeze(1)
            qd_substeps = torch.clamp(qd_substeps, -v_max.unsqueeze(1), v_max.unsqueeze(1))
            q = q + integration_dt * torch.sum(qd_substeps, dim=1)
            qd = qd_substeps[:, -1, :]
            q_trajs.append(q.unsqueeze(1))
        return torch.cat(q_trajs, dim=1)

    # --------------------------------------------------------------- cost
    def safety_cost(self, q_trajs):
        """J_safe [K] of joint trajectories [K, H, 7]."""
        q_trajs = q_trajs.to(device=self.device, dtype=torch.float32)
        K, H, dq = q_trajs.shape
        seg_start_flat, seg_end_flat = self.fk.forward_collision_segments(q_trajs.reshape(-1, dq))
        S = seg_start_flat.shape[1]
        seg_start = seg_start_flat[:, :, :3].reshape(K, H, S, 3)
        seg_end = seg_end_flat[:, :, :3].reshape(K, H, S, 3)
        seg_r = self.capsule_radii[:S].view(1, 1, -1)

        # J_obs: obstacles overlapping any capsule, per state.
        J_obs = torch.zeros(K, device=self.device, dtype=torch.float32)
        for obs in self.obstacles:
            if obs["shape"] == "sphere":
                center = obs["center"].to(device=self.device, dtype=torch.float32)
                seg = seg_end - seg_start
                seg_len2 = torch.sum(seg * seg, dim=-1, keepdim=True).clamp_min(1e-8)
                rel = center.view(1, 1, 1, 3) - seg_start
                proj = torch.clamp(torch.sum(rel * seg, dim=-1, keepdim=True) / seg_len2, 0.0, 1.0)
                closest = seg_start + proj * seg
                dist = torch.norm(closest - center.view(1, 1, 1, 3), dim=-1)
                limit = obs["radius"].to(device=self.device, dtype=torch.float32) + seg_r
                hit = torch.any(torch.relu(limit - dist) > 0.0, dim=2).float()
            else:
                center = obs["center"].to(device=self.device, dtype=torch.float32).view(1, 3)
                quat = obs["quat"].to(device=self.device, dtype=torch.float32).view(1, 4)
                half = obs["half_extents"].to(device=self.device, dtype=torch.float32)
                rot = _torch_quat_to_rotmat_wxyz(quat)[0]
                local_start = (seg_start.reshape(-1, 3) - center) @ rot
                local_end = (seg_end.reshape(-1, 3) - center) @ rot
                radii = seg_r.expand(K, H, S).reshape(-1)
                if obs["shape"] == "box":
                    overlap = _torch_capsule_box_overlap_analytic(local_start, local_end, radii, half)
                else:
                    overlap = _torch_capsule_tri_prism_overlap_analytic(local_start, local_end, radii, half)
                hit = torch.any(overlap.view(K, H, S), dim=2).float()
            J_obs += torch.sum(hit * self.w_obs, dim=1)

        # J_self: capsule pairs closer than the sum of their radii.
        p = self.self_pairs
        dist = _segment_segment_distance(seg_start[:, :, p[:, 0], :], seg_end[:, :, p[:, 0], :], seg_start[:, :, p[:, 1], :], seg_end[:, :, p[:, 1], :])
        self_hit = torch.any(torch.relu(self.self_pair_radii.view(1, 1, -1) - dist) > 0.0, dim=2).float()
        J_self = torch.sum(self_hit * self.w_self, dim=1)

        # J_floor: moving capsules (all but the base) reaching below the floor height.
        floor_radii = self.capsule_radii[1:S].view(1, 1, -1)
        floor_threshold = self.floor_height + floor_radii
        seg_min_z = torch.minimum(seg_start[:, :, 1:, 2], seg_end[:, :, 1:, 2])
        floor_hit = torch.any(torch.relu(floor_threshold - seg_min_z) > 0.0, dim=2).float()
        J_floor = torch.sum(floor_hit * self.w_floor, dim=1)

        # J_jl: joints outside the (shrunk) limits.
        lower = self.jl_lower[:dq].view(1, 1, dq)
        upper = self.jl_upper[:dq].view(1, 1, dq)
        violation = torch.relu(lower - q_trajs) + torch.relu(q_trajs - upper)
        jl_hit = torch.any(violation > 0.0, dim=2).float()
        J_jl = self.w_jl * torch.sum(jl_hit, dim=1)

        return J_obs + J_self + J_floor + J_jl
