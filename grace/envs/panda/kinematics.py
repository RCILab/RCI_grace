"""Forward kinematics of the arm with Pinocchio (start/goal sampling and demonstrations)."""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from grace.envs.panda.config import URDF


class PandaPinocchio:
    def __init__(self, urdf_path=URDF, ee_frame: str = "panda_hand_tcp") -> None:
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()

    def forward_kinematics(self, q: np.ndarray) -> pin.SE3:
        """Pose of the end-effector (TCP) frame."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id]
