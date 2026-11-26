"""Configuration of the FR3 goal-reaching task (paper Table I, "FR3").

The robot model is the Franka Panda/FR3 arm of MuJoCo Menagerie (torque
controlled, 1 kHz simulation, 50 Hz control). Defaults are the values used for
the paper's demonstrations and evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path, PosixPath

import numpy as np


ASSET_DIR = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSET_DIR / "xml" / "scene.xml"
URDF = ASSET_DIR / "panda.urdf"


def _vec(*values):
    return field(default_factory=lambda: np.array(values, dtype=float))


@dataclass
class PandaConfig:
    ee_frame: str = "panda_hand_tcp"
    # simulation / control
    sim_dt: float = 0.001
    control_dt: float = 0.02
    finger_q: float = 0.04  # gripper opening (fingers are not actuated)
    max_joint_velocity: np.ndarray = _vec(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
    torque_kp: np.ndarray = _vec(170.0, 150.0, 170.0, 150.0, 330.0, 330.0, 420.0)
    torque_kd: np.ndarray = _vec(50.0, 50.0, 50.0, 50.0, 30.0, 30.0, 30.0)
    # start / goal sampling
    workspace_low: np.ndarray = _vec(-0.55, -0.65, 0.05)  # end-effector position box (m)
    workspace_high: np.ndarray = _vec(0.75, 0.65, 0.9)
    min_goal_distance: float = 0.5  # minimal end-effector start-goal distance (m)
    joint_margin: float = 0.05  # sampled configurations keep this margin to the joint limits (rad)
    max_sampling_tries: int = 400
    goal_candidates_per_start: int = 48
    # demonstrations (feedback joint-goal expert)
    trajectory_duration: float = 20.0  # maximal demonstration length (s)
    episode_horizon: int | None = None  # derived: round(trajectory_duration / control_dt) + 1
    goal_joint_tolerance: float = 0.01  # a demonstration ends when max_j |q_j - q_goal_j| <= tolerance
    feedback_joint_kp: float = 2.0
    feedback_joint_kd: float = 0.5
    seed: int = 0

    def __post_init__(self):
        ratio = self.control_dt / self.sim_dt
        if not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValueError("control_dt must be an integer multiple of sim_dt")
        if self.episode_horizon is None:
            self.episode_horizon = int(round(self.trajectory_duration / self.control_dt)) + 1

    @property
    def substeps(self) -> int:
        return int(round(self.control_dt / self.sim_dt))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_saved(cls, saved) -> "PandaConfig":
        """Builds the config from the dictionary (or its repr) stored with a dataset / checkpoint.

        The paper datasets store a longer configuration (it also lists options
        that were not used, e.g. other expert or control modes); those entries
        are ignored after checking that they hold the values of the paper.
        """
        if isinstance(saved, str):
            saved = parse_config_repr(saved)
        known = {f.name for f in fields(cls)}
        unused = set(saved) - known
        unexpected = unused - set(_PAPER_ONLY_FIELDS)
        if unexpected:
            raise ValueError("Unknown Panda config fields: {}".format(sorted(unexpected)))
        for name in unused:
            expected = _PAPER_ONLY_FIELDS[name]
            if expected is not None and str(saved[name]) != str(expected):
                raise ValueError("Unsupported value {}={!r} (the release implements {!r})".format(name, saved[name], expected))
        return cls(**{k: v for k, v in saved.items() if k in known})


def parse_config_repr(text: str) -> dict:
    """Parses ``str(dict)`` of a configuration (only PosixPath and numpy arrays are allowed)."""
    allowed = {"__builtins__": {}, "PosixPath": PosixPath, "array": np.array}
    parsed = eval(text, allowed, {})  # noqa: S307 - restricted namespace, trusted local files
    if not isinstance(parsed, dict):
        raise ValueError("saved config is not a dictionary")
    return parsed


# Fields of the paper's dataset configuration that the release does not need.
# Value None: any value is accepted (unused by the task); otherwise the value
# must match, since the release implements only that option.
_PAPER_ONLY_FIELDS = {
    "xml_path": "franka_description/robots/panda/xml/scene.xml",
    "urdf_path": None,
    "package_dirs": None,
    "ik_max_iters": None,
    "ik_tol": None,
    "ik_damping": None,
    "goal_local_delta_low": None,
    "goal_local_delta_high": None,
    "global_goal_sample_ratio": None,
    "home_q": None,
    "velocity_kp": None,
    "velocity_servo_kv": None,
    "goal_tolerance": None,
    "goal_orientation_tolerance": None,
    "human_render_speed": None,
    "human_hold_seconds": None,
    "render_width": None,
    "render_height": None,
    "cartesian_precheck_waypoints": None,
    "cartesian_joint_goal_delta": None,
    "reference_mode": "joint",
    "expert_mode": "feedback_joint_goal",
}
