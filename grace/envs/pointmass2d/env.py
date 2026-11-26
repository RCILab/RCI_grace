"""Planar point-mass navigation environment (paper Sec. IV-A, Fig. 3a).

State x = position in [-1, 1]^2, action v = velocity command (m/s). A step
moves the agent by clip(v) * dt; if the swept segment hits an obstacle (or the
workspace boundary) the agent stays in place and the episode ends with a
collision failure. The episode succeeds when the agent is within
``goal_tolerance`` of the goal.

Observations are dictionaries with the keys used by the diffusion prior
(``position``, ``velocity``, ``goal_delta``) plus ``goal`` and ``flat``.
"""

import numpy as np

from grace.envs.pointmass2d.config import ENV_IDS, PointMassConfig
from grace.envs.pointmass2d.geometry import points_in_collision, segment_in_collision
from grace.envs.pointmass2d.maps import make_dense2d_map


def build_observation_dict(position, velocity, goal):
    position = np.asarray(position, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    goal_delta = goal - position
    flat = np.concatenate([position, velocity, goal, goal_delta], axis=0).astype(np.float32)
    return {
        "position": position.copy(),
        "velocity": velocity.copy(),
        "goal": goal.copy(),
        "goal_delta": goal_delta.astype(np.float32),
        "flat": flat,
    }


def compute_reward(prev_position, next_position, goal, success, collision, config):
    if collision:
        return float(config.reward_collision)
    prev_dist = np.linalg.norm(np.asarray(goal) - np.asarray(prev_position))
    next_dist = np.linalg.norm(np.asarray(goal) - np.asarray(next_position))
    reward = config.reward_progress_scale * float(prev_dist - next_dist)
    reward -= config.reward_step_penalty
    if success:
        reward += config.reward_success
    return float(reward)


class PointMassEnv:
    def __init__(self, config=None, use_extra_obstacles=False, seed=None):
        self.config = config or PointMassConfig()
        self.use_extra_obstacles = bool(use_extra_obstacles)
        self.map = make_dense2d_map()
        # Start/goal sampling has its own RNG, so the episode sequence depends
        # only on the seed and the episode index (not on the policy).
        self._np_random = np.random.RandomState(seed)
        self.position = None
        self.velocity = None
        self.goal = None
        self.t = 0
        self.last_collision = False
        self.last_success = False

    def seed(self, seed=None):
        self._np_random = np.random.RandomState(seed)

    # ----------------------------------------------------------------- reset
    def _sample_collision_free_point(self, max_tries=4096):
        lower, upper = self.map.limits[0], self.map.limits[1]
        for _ in range(max_tries):
            sample = self._np_random.uniform(low=lower, high=upper, size=(2,)).astype(np.float32)
            in_collision = points_in_collision(
                sample[None, :],
                obstacle_map=self.map,
                robot_radius=self.config.agent_radius,
                clearance=0.0,
                include_extra=self.use_extra_obstacles,
            )[0]
            if not bool(in_collision):
                return sample
        raise RuntimeError("Could not sample collision-free point")

    def _sample_start_goal(self, max_tries=4096):
        for _ in range(max_tries):
            start = self._sample_collision_free_point(max_tries=64)
            goal = self._sample_collision_free_point(max_tries=64)
            if np.linalg.norm(goal - start) >= self.config.min_start_goal_distance:
                return start, goal
        raise RuntimeError("Could not sample collision-free start and goal")

    def reset(self):
        self.position, self.goal = self._sample_start_goal()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.t = 0
        self.last_collision = False
        self.last_success = False
        return self.observation()

    def observation(self):
        return build_observation_dict(self.position, self.velocity, self.goal)

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -self.config.max_speed, self.config.max_speed)

        prev_position = self.position.copy()
        next_velocity = action.astype(np.float32)
        proposed_position = prev_position + next_velocity * self.config.dt

        collision = segment_in_collision(
            prev_position,
            proposed_position,
            obstacle_map=self.map,
            robot_radius=self.config.agent_radius,
            clearance=0.0,
            include_extra=self.use_extra_obstacles,
            step_size=self.config.collision_check_step,
        )
        if collision:
            next_position = prev_position
            next_velocity = np.zeros(2, dtype=np.float32)
        else:
            next_position = proposed_position.astype(np.float32)

        success = (not collision) and (np.linalg.norm(next_position - self.goal) <= self.config.goal_tolerance)
        reward = compute_reward(prev_position, next_position, self.goal, success, collision, self.config)

        self.position = next_position
        self.velocity = next_velocity
        self.t += 1
        self.last_collision = bool(collision)
        self.last_success = bool(success)

        done = bool(collision or success or (self.t >= self.config.max_episode_steps))
        info = {
            "collision": bool(collision),
            "success": bool(success),
            "goal_distance": float(np.linalg.norm(self.goal - self.position)),
            "step": int(self.t),
        }
        return self.observation(), reward, done, info

    # ----------------------------------------------------------------- state
    def get_state(self):
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "goal": self.goal.copy(),
            "step": int(self.t),
        }

    def set_state(self, state):
        self.position = np.asarray(state["position"], dtype=np.float32).copy()
        self.velocity = np.asarray(state["velocity"], dtype=np.float32).copy()
        self.goal = np.asarray(state["goal"], dtype=np.float32).copy()
        self.t = int(state["step"])
        self.last_collision = False
        self.last_success = bool(np.linalg.norm(self.goal - self.position) <= self.config.goal_tolerance)


def make_env(env_name, env_kwargs=None, seed=None):
    """Builds the environment from the (env_name, env_kwargs) pair stored in datasets/checkpoints."""
    if env_name not in ENV_IDS:
        raise ValueError("Unknown point-mass env id '{}'; expected one of {}".format(env_name, sorted(ENV_IDS)))
    config = PointMassConfig.from_env_kwargs(env_kwargs or {})
    return PointMassEnv(config=config, use_extra_obstacles=ENV_IDS[env_name], seed=seed)
