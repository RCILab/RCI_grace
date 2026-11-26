"""Rollout model and cost of the point-mass task for GRACE (paper Sec. IV-A, eqs. (30)-(31)).

Rollout: x_{t+1} = x_t + clip(v_t, -v_max, v_max) * dt, from the current
position x_0 for the H guided actions of a candidate (same dynamics as the
environment, without the collision stop).

Cost: J = w_obs * J_obs + w_prior * J_prior (eq. (31)); this module provides
w_obs * J_obs, the prior term is added in ``grace_posterior_mean``. The goal is
not part of the cost: it is supplied by the prior.

    J_obs = sum_{t=1..H} ( 1_obs(x_t) + sum_o (p_o(x_t) / ell_o)^2 )      (eq. (30))

with the agent a disc of radius r = agent_radius + planning_clearance, and for
each obstacle o:
    disc of radius r_o:  d_o = |x - c_o| - (r_o + r),  ell_o = r_o + r
    box:                 d_o = sd_box(x) - r,           ell_o = r
    (sd_box: exact signed distance to the box)
    p_o = max(0, -d_o)   (penetration depth)
1_obs(x_t) counts the obstacles with p_o > 0. The workspace boundary
[-1 + r, 1 - r]^2 is treated as one more obstacle with ell = 1.
All obstacles are included (the base and the extra, unseen ones).
"""

import torch


class PointMassGuidanceTask:
    def __init__(self, env, w_obs, normalizer, device):
        cfg = env.config
        obstacle_map = env.map
        include_extra = env.use_extra_obstacles
        self.normalizer = normalizer
        self.device = device
        self.w_obs = float(w_obs)
        self.dt = float(cfg.dt)
        self.v_min = torch.as_tensor(-float(cfg.max_speed), dtype=torch.float32, device=device).view(-1).repeat(2)
        self.v_max = torch.as_tensor(float(cfg.max_speed), dtype=torch.float32, device=device).view(-1).repeat(2)
        self.r = float(cfg.agent_radius) + float(cfg.planning_clearance)

        def f32(value):
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        self.limits = f32(obstacle_map.limits).view(2, 2)
        self.disc_centers = f32(obstacle_map.all_sphere_centers(include_extra=include_extra)).view(-1, 2)
        self.disc_radii = f32(obstacle_map.all_sphere_radii(include_extra=include_extra)).view(-1)
        self.box_centers = f32(obstacle_map.all_box_centers(include_extra=include_extra)).view(-1, 2)
        self.box_sizes = f32(obstacle_map.all_box_sizes(include_extra=include_extra)).view(-1, 2)

    def bind(self, obs):
        """Returns V -> w_obs * J_obs for the current (normalized) observation."""
        x0 = self.normalizer.unnormalize_obs("position", obs["position"][:, -1, :]).view(-1)

        def rollout_cost(V):
            return self.w_obs * torch.sum(self.obstacle_cost_per_step(self.rollout(x0, V)), dim=1)

        return rollout_cost

    def rollout(self, x0, V):
        """x0: [2] current position, V: [K, H, 2] velocity commands -> states x_1..x_H [K, H, 2]."""
        V = torch.as_tensor(V, dtype=torch.float32, device=self.device)
        m = V.shape[-1]
        V = torch.maximum(V, self.v_min.view(1, 1, m))
        V = torch.minimum(V, self.v_max.view(1, 1, m))
        K = V.shape[0]
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)[:, :m]
        return x0.repeat(K, 1).unsqueeze(1) + torch.cumsum(V * self.dt, dim=1)

    def obstacle_cost_per_step(self, X):
        """1_obs(x_t) + sum_o (p_o(x_t) / ell_o)^2 for every state, [K, H]."""
        xy = X[..., :2].to(dtype=torch.float32)
        device = xy.device
        K, H, _ = xy.shape
        r = self.r
        indicator = torch.zeros((K, H), dtype=torch.float32, device=device)
        penalty = torch.zeros((K, H), dtype=torch.float32, device=device)

        # Workspace boundary.
        lower = self.limits[0].view(1, 1, 2) + r
        upper = self.limits[1].view(1, 1, 2) - r
        p_bound = torch.relu(lower - xy) + torch.relu(xy - upper)
        indicator = indicator + torch.any(p_bound > 0.0, dim=-1).float()
        penalty = penalty + torch.sum(p_bound * p_bound, dim=-1)

        # Discs: ell_o = r_o + r.
        if self.disc_centers.numel() > 0:
            centers = self.disc_centers.view(1, 1, -1, 2)
            ell = (self.disc_radii.view(1, 1, -1) + r).clamp_min(1e-6)
            dist = torch.norm(xy.unsqueeze(2) - centers, dim=-1)
            p = torch.relu(ell - dist)
            indicator = indicator + torch.sum((p > 0.0).float(), dim=2)
            penalty = penalty + torch.sum((p / ell) ** 2, dim=2)

        # Boxes (exact signed distance): ell_o = r.
        if self.box_centers.numel() > 0:
            centers = self.box_centers.view(1, 1, -1, 2)
            half_sizes = (0.5 * self.box_sizes).view(1, 1, -1, 2)
            delta = torch.abs(xy.unsqueeze(2) - centers) - half_sizes
            outside_dist = torch.norm(torch.relu(delta), dim=-1)
            inside_dist = torch.minimum(
                torch.amax(delta, dim=-1),
                torch.zeros((), dtype=torch.float32, device=device),
            )
            sd_box = outside_dist + inside_dist
            ell = torch.tensor(max(r, 1e-6), dtype=torch.float32, device=device)
            p = torch.relu(ell - sd_box)
            indicator = indicator + torch.sum((p > 0.0).float(), dim=2)
            penalty = penalty + torch.sum((p / ell) ** 2, dim=2)

        return indicator + penalty
