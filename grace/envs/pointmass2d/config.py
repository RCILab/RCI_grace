"""Configuration of the planar point-mass task (paper Table I, "Point-mass").

Defaults are the values used in the paper. The environment id stored in the
dataset / checkpoint metadata selects the obstacle set:

    PointMassDense2D-v0       base obstacles only (demonstrations, training)
    PointMassDense2DExtra-v0  base + extra obstacles (evaluation, Fig. 3a)
"""

from dataclasses import asdict, dataclass, fields


ENV_ID_TRAIN = "PointMassDense2D-v0"
ENV_ID_EVAL = "PointMassDense2DExtra-v0"
ENV_IDS = {ENV_ID_TRAIN: False, ENV_ID_EVAL: True}  # env id -> use_extra_obstacles

# Keys found in the env_kwargs of the paper datasets/checkpoints that belong to
# options which were not used for the paper (other maps, other planners).
_UNUSED_ENV_KWARGS = ("narrow_passage_side_margin", "planner_max_iters", "planner_step_size")


@dataclass(frozen=True)
class PointMassConfig:
    # dynamics: x_{t+1} = x_t + clip(v_t, -max_speed, max_speed) * dt
    dt: float = 0.02
    max_speed: float = 0.5
    # geometry
    agent_radius: float = 0.02
    planning_clearance: float = 0.005  # extra margin used by the demo planner and the GRACE cost
    collision_check_step: float = 0.01  # spacing of the swept collision check per step
    # task
    goal_tolerance: float = 0.05
    max_episode_steps: int = 1000
    min_start_goal_distance: float = 1.0
    # reward (stored in the dataset; not used by the method)
    reward_step_penalty: float = 0.01
    reward_progress_scale: float = 1.0
    reward_success: float = 1.0
    reward_collision: float = -1.0
    # demonstration planner (OMPL RRTConnect)
    planner_radius: float = 0.3
    planner_max_time: float = 50.0
    planner_num_pre_samples: int = 50000
    planner_shortcut_iterations: int = 128
    stop_action_steps: int = 1

    def env_kwargs(self):
        return asdict(self)

    @classmethod
    def from_env_kwargs(cls, env_kwargs):
        known = {f.name for f in fields(cls)}
        unknown = set(env_kwargs) - known - set(_UNUSED_ENV_KWARGS)
        if unknown:
            raise ValueError("Unknown point-mass env kwargs: {}".format(sorted(unknown)))
        return cls(**{k: v for k, v in env_kwargs.items() if k in known})
