"""Expert demonstrations for the point-mass task: OMPL RRTConnect paths converted to velocity commands.

Demonstrations are collected on the base map only (no extra obstacles) with a
clearance of ``planning_clearance`` around the agent.
"""

import numpy as np

from grace.envs.pointmass2d.geometry import points_in_collision, resample_path_chebyshev, shortcut_path


class OMPLPointMassPlanner:
    """RRTConnect (OMPL) in the 2D workspace, followed by path simplification and shortcutting."""

    def __init__(self, obstacle_map, config, include_extra=False):
        try:
            from ompl import base as ob
            from ompl import geometric as og
        except ImportError as exc:
            raise ImportError("Demonstration generation requires the OMPL Python bindings (pip install ompl)") from exc
        self.ob = ob
        self.og = og
        self.map = obstacle_map
        self.config = config
        self.include_extra = include_extra

    def _in_collision(self, points):
        return points_in_collision(
            points,
            obstacle_map=self.map,
            robot_radius=self.config.agent_radius,
            clearance=self.config.planning_clearance,
            include_extra=self.include_extra,
        )

    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if bool(self._in_collision(np.stack([start, goal], axis=0)).any()):
            return None

        space = self.ob.RealVectorStateSpace(2)
        bounds = self.ob.RealVectorBounds(2)
        bounds.setLow(0, float(self.map.limits[0, 0]))
        bounds.setLow(1, float(self.map.limits[0, 1]))
        bounds.setHigh(0, float(self.map.limits[1, 0]))
        bounds.setHigh(1, float(self.map.limits[1, 1]))
        space.setBounds(bounds)

        def make_state(point):
            state = space.allocState()
            state[0] = float(point[0])
            state[1] = float(point[1])
            return state

        simple_setup = self.og.SimpleSetup(space)

        def is_state_valid(state):
            point = np.asarray([state[0], state[1]], dtype=np.float32)
            return not bool(self._in_collision(point[None, :])[0])

        simple_setup.setStateValidityChecker(is_state_valid)
        si = simple_setup.getSpaceInformation()
        maximum_extent = float(space.getMaximumExtent())
        validity_resolution = self.config.collision_check_step / max(maximum_extent, 1e-6)
        validity_resolution = min(max(validity_resolution, 1e-4), 0.25)
        try:
            si.setStateValidityCheckingResolution(validity_resolution)
        except Exception:
            pass

        planner = self.og.RRTConnect(si)
        if hasattr(planner, "setRange"):
            planner.setRange(float(self.config.planner_radius))
        simple_setup.setPlanner(planner)
        simple_setup.setStartAndGoalStates(make_state(start), make_state(goal))
        if not simple_setup.solve(float(self.config.planner_max_time)):
            return None
        try:
            simple_setup.simplifySolution()
        except Exception:
            pass

        solution = simple_setup.getSolutionPath()
        path = np.asarray(
            [[float(solution.getState(i)[0]), float(solution.getState(i)[1])] for i in range(int(solution.getStateCount()))],
            dtype=np.float32,
        )
        if path.shape[0] <= 1:
            return None
        if float(np.linalg.norm(path[-1] - goal)) > float(self.config.goal_tolerance):
            return None
        return shortcut_path(
            path,
            obstacle_map=self.map,
            robot_radius=self.config.agent_radius,
            clearance=self.config.planning_clearance,
            include_extra=self.include_extra,
            step_size=self.config.collision_check_step,
            n_iterations=self.config.planner_shortcut_iterations,
        )


class PointMassExpert:
    """Samples start/goal pairs and converts planned paths into velocity-command demonstrations."""

    def __init__(self, obstacle_map, config, rng):
        self.map = obstacle_map
        self.config = config
        self.rng = rng
        self.planner = OMPLPointMassPlanner(obstacle_map, config, include_extra=False)
        # The paper datasets were generated with a planner implementation that
        # drew a pool of collision-free samples from ``rng`` at construction
        # (it is not used by OMPL). The draw is kept so that the same seed
        # yields the same sequence of start/goal pairs.
        self.sample_collision_free(
            n_samples=self.config.planner_num_pre_samples,
            max_tries=max(self.config.planner_num_pre_samples // 4, 256),
        )

    def _in_collision(self, points):
        return points_in_collision(
            points,
            obstacle_map=self.map,
            robot_radius=self.config.agent_radius,
            clearance=self.config.planning_clearance,
            include_extra=False,
        )

    def sample_collision_free(self, n_samples=1, max_tries=1024):
        samples = []
        lower, upper = self.map.limits[0], self.map.limits[1]
        for _ in range(max_tries):
            batch = self.rng.uniform(low=lower, high=upper, size=(max(n_samples, 64), 2)).astype(np.float32)
            free_mask = ~self._in_collision(batch)
            if np.any(free_mask):
                samples.extend(batch[free_mask].tolist())
            if len(samples) >= n_samples:
                return np.asarray(samples[:n_samples], dtype=np.float32)
        raise RuntimeError("Could not sample enough collision-free points")

    def sample_start_goal(self, max_tries=2048):
        for _ in range(max_tries):
            start, goal = self.sample_collision_free(n_samples=2, max_tries=64)
            if np.linalg.norm(goal - start) >= self.config.min_start_goal_distance:
                return start, goal
        raise RuntimeError("Could not sample a valid start-goal pair")

    def plan(self, start, goal):
        return self.planner.plan(start, goal)

    def path_to_actions(self, path):
        """Resamples the path at the maximal speed and returns the velocity commands, or None.

        Each step moves at most dt * max_speed along either axis, and
        ``stop_action_steps`` zero commands are appended (they count towards
        ``max_episode_steps``). The replay in ``generate.rollout_demo`` ends as
        soon as the goal tolerance is reached, which happens before the zero
        commands, so the stored demonstrations end with a motion command.
        """
        cfg = self.config
        positions = resample_path_chebyshev(np.asarray(path, dtype=np.float32), max_delta=cfg.dt * cfg.max_speed)
        if positions.shape[0] < 2:
            return None
        if (positions.shape[0] - 1) + cfg.stop_action_steps > cfg.max_episode_steps:
            return None
        actions = [
            np.clip((nxt - cur) / cfg.dt, -cfg.max_speed, cfg.max_speed).astype(np.float32)
            for cur, nxt in zip(positions[:-1], positions[1:])
        ]
        actions += [np.zeros(2, dtype=np.float32) for _ in range(cfg.stop_action_steps)]
        return np.asarray(actions, dtype=np.float32)
