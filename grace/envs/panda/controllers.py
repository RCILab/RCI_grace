"""Joint-space torque tracking controller of the simulated arm (also used by the GRACE rollout model)."""

from __future__ import annotations

import mujoco
import numpy as np


def clip_joint_velocity(qd_des: np.ndarray, max_joint_velocity: np.ndarray) -> np.ndarray:
    return np.clip(qd_des, -max_joint_velocity, max_joint_velocity)


def compute_torque_tracking_qacc_des(q, qd, q_des, qd_des, kp, kd):
    """Desired joint acceleration kp (q_des - q) + kd (qd_des - qd) (numpy or torch)."""
    return kp * (q_des - q) + kd * (qd_des - qd)


class TorqueTrackingController:
    """tau = M(q) qacc_des + bias(q, qd), with M and bias from the last ``mj_forward``."""

    def __init__(self, model: mujoco.MjModel, kp: np.ndarray, kd: np.ndarray) -> None:
        self.model = model
        self.kp = np.asarray(kp, dtype=float).reshape(-1)
        self.kd = np.asarray(kd, dtype=float).reshape(-1)

    def compute(self, data, arm_qpos_idx, arm_qvel_idx, q_des, qd_des) -> np.ndarray:
        q = data.qpos[arm_qpos_idx].copy()
        qd = data.qvel[arm_qvel_idx].copy()
        qacc_des = compute_torque_tracking_qacc_des(q=q, qd=qd, q_des=q_des, qd_des=qd_des, kp=self.kp, kd=self.kd)
        mass = np.zeros((self.model.nv, self.model.nv), dtype=float)
        mujoco.mj_fullM(self.model, mass, data.qM)
        bias = data.qfrc_bias.copy()
        return mass[np.ix_(arm_qvel_idx, arm_qvel_idx)] @ qacc_des + bias[arm_qvel_idx]
