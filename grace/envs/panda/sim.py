"""MuJoCo simulation of the torque-controlled arm.

Two variants reproduce the two pipelines of the paper exactly:

``PandaDemoSim`` (demonstrations): the passive fingers are pinned open after
every physics substep, and ``mj_forward`` is called right before each control
step, so the torque law uses the mass matrix / bias forces of the current state.

``PandaEvalSim`` (evaluation): the fingers are passive after the reset, the
torque command is held for the 20 substeps of a control period and
``mj_forward`` is called after them (contacts and dynamics quantities are then
current for the failure checks and the next command). One MjData is used for
all trials of an evaluation run.

The obstacles of the evaluation are not physical in the simulation; robot-
obstacle contact is checked separately (``grace.eval.panda.ObstacleChecker``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from grace.envs.panda.config import SCENE_XML
from grace.envs.panda.controllers import TorqueTrackingController, clip_joint_velocity

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=float)  # TCP in the panda_hand frame


class _PandaSim:
    def __init__(self, config, xml_path=SCENE_XML):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = config.sim_dt
        self.data = mujoco.MjData(self.model)
        m = self.model

        def joint_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

        self.arm_qpos_idx = np.array([m.jnt_qposadr[joint_id(n)] for n in ARM_JOINTS], dtype=int)
        self.arm_qvel_idx = np.array([m.jnt_dofadr[joint_id(n)] for n in ARM_JOINTS], dtype=int)
        self.arm_act_idx = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINTS], dtype=int)
        self.finger_qpos_idx = np.array([m.jnt_qposadr[joint_id(n)] for n in FINGER_JOINTS], dtype=int)
        self.finger_qvel_idx = np.array([m.jnt_dofadr[joint_id(n)] for n in FINGER_JOINTS], dtype=int)
        self.ee_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        self.floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.robot_body_ids = {
            body_id
            for body_id in range(m.nbody)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith("panda_")
        }
        self.controller = TorqueTrackingController(model=m, kp=config.torque_kp, kd=config.torque_kd)

    # ---------------------------------------------------------------- state
    def _set_configuration(self, q):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[self.arm_qpos_idx] = q
        self.data.qpos[self.finger_qpos_idx] = self.config.finger_q

    def observation(self):
        """q, qd, end-effector (TCP) position and orientation (xyzw)."""
        ee_rot = self.data.xmat[self.ee_body_id].reshape(3, 3)
        ee_pos = self.data.xpos[self.ee_body_id] + ee_rot @ TCP_OFFSET
        return {
            "q": self.data.qpos[self.arm_qpos_idx].copy(),
            "qd": self.data.qvel[self.arm_qvel_idx].copy(),
            "ee_pos": ee_pos.copy(),
            "ee_quat_xyzw": Rotation.from_matrix(ee_rot).as_quat(),
        }

    # ------------------------------------------------------------ contacts
    def _is_robot_geom(self, geom_id):
        return geom_id >= 0 and int(self.model.geom_bodyid[geom_id]) in self.robot_body_ids

    def collision_flags(self):
        """(self_collision, floor_collision) from the current MuJoCo contacts."""
        self_collision = False
        floor_collision = False
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            robot1, robot2 = self._is_robot_geom(geom1), self._is_robot_geom(geom2)
            if robot1 and robot2:
                self_collision = True
            if (geom1 == self.floor_geom_id and robot2) or (geom2 == self.floor_geom_id and robot1):
                floor_collision = True
        return self_collision, floor_collision

    def collision_flags_for_q(self, q):
        """Self / floor contact of a static configuration (used when sampling start and goal)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (7,):
            raise ValueError(f"Expected q shape (7,), got {q.shape}")
        self._set_configuration(q)
        mujoco.mj_forward(self.model, self.data)
        return self.collision_flags()


@dataclass
class DemoStepResult:
    obs: dict
    tau: np.ndarray
    self_collision: bool
    floor_collision: bool


class PandaDemoSim(_PandaSim):
    """Simulation used to record the demonstrations."""

    def __init__(self, config, xml_path=SCENE_XML):
        super().__init__(config, xml_path)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, q):
        self._set_configuration(q)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def step(self, q_des, qd_des):
        mujoco.mj_forward(self.model, self.data)
        qd_des = clip_joint_velocity(qd_des, self.config.max_joint_velocity)
        tau = self.controller.compute(self.data, self.arm_qpos_idx, self.arm_qvel_idx, q_des=q_des, qd_des=qd_des)
        self.data.ctrl[self.arm_act_idx] = tau
        self_collision = False
        floor_collision = False
        for _ in range(self.config.substeps):
            mujoco.mj_step(self.model, self.data)
            # The gripper is not used: pin the passive fingers open.
            self.data.qpos[self.finger_qpos_idx] = float(self.config.finger_q)
            self.data.qvel[self.finger_qvel_idx] = 0.0
            step_self, step_floor = self.collision_flags()
            self_collision = self_collision or step_self
            floor_collision = floor_collision or step_floor
        return DemoStepResult(self.observation(), tau, self_collision, floor_collision)


class PandaEvalSim(_PandaSim):
    """Simulation used for the closed-loop evaluation."""

    def reset(self, q):
        self._set_configuration(q)
        self.data.qvel[self.finger_qvel_idx] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def torque_for_command(self, q_des, qd_des):
        return self.controller.compute(self.data, self.arm_qpos_idx, self.arm_qvel_idx, q_des=q_des, qd_des=qd_des)

    def gravity_hold_torque(self):
        """Holds the current configuration (q_des = q, qd_des = 0)."""
        q_hold = self.data.qpos[self.arm_qpos_idx].copy()
        return self.torque_for_command(q_hold, np.zeros_like(q_hold))

    def step(self, tau):
        """Applies ``tau`` for one control period."""
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.arm_act_idx] = np.asarray(tau, dtype=float).reshape(-1)
        for _ in range(self.config.substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
