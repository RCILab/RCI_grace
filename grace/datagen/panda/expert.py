"""Obstacle-free demonstrations of the FR3 task: a feedback joint-goal expert (paper Table I, "FR3").

Each control step, the expert commands the joint target

    qd_des = clip(kp_fb (q_goal - q) - kd_fb qd, +-v_max)
    q_des  = clip(q + qd_des dt, joint limits - margin),   qd_des = (q_des - q) / dt

which the torque controller of the simulation tracks. A demonstration ends
when max_j |q_j - q_goal_j| <= goal_joint_tolerance (or after
``trajectory_duration``) and is accepted if it reached the goal without self or
floor contact. The policy action is delta_q = q_des - q (see ``convert``).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from grace.envs.panda.kinematics import PandaPinocchio
from grace.envs.panda.sampling import sample_episode_endpoints
from grace.envs.panda.sim import PandaDemoSim


def feedback_joint_goal_command(q, qd, q_goal, joint_lower, joint_upper, config):
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    qd_des = config.feedback_joint_kp * (q_goal - q) - config.feedback_joint_kd * qd
    qd_des = np.clip(qd_des, -config.max_joint_velocity, config.max_joint_velocity)
    q_des = np.clip(q + qd_des * float(config.control_dt), joint_lower, joint_upper)
    qd_des = (q_des - q) / float(config.control_dt)
    return q_des, qd_des


def generate_episode(rng, config):
    """Samples a start/goal pair from ``rng`` and records one demonstration."""
    kin = PandaPinocchio(ee_frame=config.ee_frame)
    sim = PandaDemoSim(config)
    start_pos, start_rot, q_start, goal_pos, goal_rot, q_goal = sample_episode_endpoints(rng, kin, config, collision_checker=sim)

    joint_lower = kin.lower + config.joint_margin
    joint_upper = kin.upper - config.joint_margin
    obs = sim.reset(q_start)

    horizon = int(config.episode_horizon)
    obs_q = np.zeros((horizon, 7))
    obs_qd = np.zeros((horizon, 7))
    obs_ee = np.zeros((horizon, 3))
    obs_ee_quat = np.zeros((horizon, 4))
    desired_q = np.zeros((horizon, 7))
    desired_qd = np.zeros((horizon, 7))
    tau = np.zeros((horizon, 7))
    had_self_collision = False
    had_floor_collision = False
    step = 0
    for _ in range(horizon):
        # The observation is recorded at decision time (the state the command is computed from).
        obs_q[step] = obs["q"]
        obs_qd[step] = obs["qd"]
        obs_ee[step] = obs["ee_pos"]
        obs_ee_quat[step] = obs["ee_quat_xyzw"]
        q_des, qd_des = feedback_joint_goal_command(obs["q"], obs["qd"], q_goal, joint_lower, joint_upper, config)
        desired_q[step] = q_des
        desired_qd[step] = qd_des
        result = sim.step(q_des, qd_des)
        tau[step] = result.tau
        obs = result.obs
        had_self_collision = had_self_collision or result.self_collision
        had_floor_collision = had_floor_collision or result.floor_collision
        step += 1
        if had_self_collision or had_floor_collision:
            break
        if float(np.max(np.abs(result.obs["q"] - q_goal))) <= config.goal_joint_tolerance:
            break
    n = max(step, 1)

    final_joint_error_max = float(np.max(np.abs(np.asarray(obs["q"], dtype=float) - q_goal)))
    goal_reached = bool(
        (not had_self_collision) and (not had_floor_collision) and final_joint_error_max <= config.goal_joint_tolerance
    )
    return {
        "timestamps": (np.arange(horizon, dtype=float) * float(config.control_dt))[:n].copy(),
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "start_quat_xyzw": Rotation.from_matrix(start_rot).as_quat(),
        "goal_quat_xyzw": Rotation.from_matrix(goal_rot).as_quat(),
        "q_start": q_start,
        "q_goal": q_goal,
        "goal_distance": np.array(float(np.linalg.norm(goal_pos - start_pos)), dtype=float),
        "observed_q": obs_q[:n].copy(),
        "observed_qd": obs_qd[:n].copy(),
        "observed_ee_pos": obs_ee[:n].copy(),
        "observed_ee_quat_xyzw": obs_ee_quat[:n].copy(),
        "desired_q": desired_q[:n].copy(),
        "desired_qd": desired_qd[:n].copy(),
        "applied_tau": tau[:n].copy(),
        "final_ee_error": np.array(float(np.linalg.norm(np.asarray(obs["ee_pos"], dtype=float) - goal_pos)), dtype=float),
        "final_joint_error_max": np.array(final_joint_error_max, dtype=float),
        "had_self_collision": np.array(had_self_collision, dtype=bool),
        "had_floor_collision": np.array(had_floor_collision, dtype=bool),
        "goal_reached": np.array(goal_reached, dtype=bool),
    }
