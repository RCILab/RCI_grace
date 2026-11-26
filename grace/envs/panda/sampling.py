"""Random start and goal configurations (shared by the demonstrations and the evaluation).

A configuration is drawn uniformly within the joint limits (minus
``joint_margin``) and accepted if its end-effector lies in the workspace box and
the arm is free of self and floor contact. A goal is accepted if its
end-effector is at least ``min_goal_distance`` away from the start's. The
number and order of random draws define the episodes for a given seed.
"""

from __future__ import annotations

import numpy as np


def sample_workspace_configuration(rng, kin, config, collision_checker=None, num_tries=None):
    """Returns (ee_pos, ee_rotation, q) of a valid configuration, or raises RuntimeError."""
    num_tries = num_tries or config.max_sampling_tries
    lower = kin.lower + config.joint_margin
    upper = kin.upper - config.joint_margin
    for _ in range(num_tries):
        q = rng.uniform(lower, upper)
        pose = kin.forward_kinematics(q)
        pos = pose.translation.copy()
        if not (np.all(pos >= config.workspace_low) and np.all(pos <= config.workspace_high)):
            continue
        if collision_checker is not None:
            self_collision, floor_collision = collision_checker.collision_flags_for_q(q)
            if self_collision or floor_collision:
                continue
        return pos, pose.rotation.copy(), q
    raise RuntimeError("Failed to sample a reachable joint configuration inside the workspace.")


def sample_episode_endpoints(rng, kin, config, collision_checker=None):
    """Start and goal: (start_pos, start_rot, q_start, goal_pos, goal_rot, q_goal).

    For each start, up to ``goal_candidates_per_start`` goal candidates are
    drawn (one configuration draw each).
    """
    for _ in range(config.max_sampling_tries):
        start_pos, start_rot, q_start = sample_workspace_configuration(rng, kin, config, collision_checker=collision_checker)
        for _ in range(config.goal_candidates_per_start):
            try:
                goal_pos, goal_rot, q_goal = sample_workspace_configuration(
                    rng, kin, config, collision_checker=collision_checker, num_tries=1
                )
            except RuntimeError:
                continue
            if float(np.linalg.norm(goal_pos - start_pos)) < config.min_goal_distance:
                continue
            return start_pos, start_rot, q_start, goal_pos, goal_rot, q_goal
    raise RuntimeError("Failed to sample a reachable start/goal pair.")
