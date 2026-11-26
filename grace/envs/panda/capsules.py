"""Capsule model of the arm, derived from the MuJoCo collision geometry.

Every collision geom of the arm (links 0-7, hand, fingers) is replaced by a
capsule (segment + radius) attached to its body. Batched forward kinematics
(torch, standard Panda DH parameters) gives the world-frame segments. The
capsules are used by the GRACE cost (obstacle, self and floor terms) and by the
conservative checks of the evaluation's start/goal sampling.
"""

from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np
import torch

from grace.envs.panda.config import SCENE_XML

_COLLISION_BODIES = {
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_left_finger",
    "panda_right_finger",
}
_GRIPPER_BODIES = {"panda_hand", "panda_left_finger", "panda_right_finger"}


def _quat_wxyz_to_rotmat(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.asarray(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=float,
    )


def _capsule_from_geom(geom_type: int, geom_pos, geom_quat, geom_size):
    """Segment (start, end) in the body frame and radius of the capsule enclosing a geom."""
    geom_pos = np.asarray(geom_pos, dtype=float).reshape(3)
    geom_quat = np.asarray(geom_quat, dtype=float).reshape(4)
    geom_size = np.asarray(geom_size, dtype=float).reshape(3)
    R = _quat_wxyz_to_rotmat(geom_quat)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return geom_pos, geom_pos, float(geom_size[0])

    if geom_type in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        radius = float(geom_size[0])
        half_length = float(max(geom_size[1], 0.0))
        axis = R[:, 2]
        return geom_pos - axis * half_length, geom_pos + axis * half_length, radius

    extents = np.maximum(geom_size[:3], 0.0)
    major_axis = int(np.argmax(extents))
    minor_axes = [idx for idx in range(3) if idx != major_axis]
    radius = float(max(extents[minor_axes[0]], extents[minor_axes[1]]))
    half_length = float(max(extents[major_axis] - radius, 0.0))
    if half_length <= 1e-6:
        return geom_pos, geom_pos, float(max(extents[major_axis], radius))
    axis = R[:, major_axis]
    return geom_pos - axis * half_length, geom_pos + axis * half_length, radius


def _geom_radius(model, geom_id):
    size = model.geom_size[geom_id]
    geom_type = int(model.geom_type[geom_id])
    if geom_type in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        return float(size[0])
    if geom_type in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_MESH)):
        return float(max(size[0], size[1]))
    return float(max(size[0], size[1], 0.0))


@lru_cache(maxsize=4)
def load_capsule_geometry(xml_path=str(SCENE_XML)):
    """Capsules, gripper capsule indices, fixed hand/finger poses and arm joint limits from the scene XML."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    def body_name(geom_id):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))

    def body_pose(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError("body {!r} not found in {}".format(name, xml_path))
        return tuple(float(v) for v in model.body_pos[bid]), tuple(float(v) for v in model.body_quat[bid])

    capsules = []
    gripper_indices = []
    for geom_id in range(model.ngeom):
        name = body_name(geom_id)
        if name not in _COLLISION_BODIES:
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        geom_radius = float(model.geom_rbound[geom_id])
        if geom_radius <= 0.0:
            geom_radius = _geom_radius(model, geom_id)
        if geom_radius <= 0.0:
            continue
        local_start, local_end, radius = _capsule_from_geom(
            int(model.geom_type[geom_id]), model.geom_pos[geom_id], model.geom_quat[geom_id], model.geom_size[geom_id]
        )
        # Grip pads lie inside the finger capsule; they would only duplicate checks.
        if name in {"panda_left_finger", "panda_right_finger"} and radius < 0.010:
            continue
        capsules.append(
            (
                str(name),
                float(local_start[0]),
                float(local_start[1]),
                float(local_start[2]),
                float(local_end[0]),
                float(local_end[1]),
                float(local_end[2]),
                float(max(radius, 1e-4)),
            )
        )
        if name in _GRIPPER_BODIES:
            gripper_indices.append(len(capsules) - 1)
    if not capsules:
        raise RuntimeError("no arm collision geoms found in {}".format(xml_path))

    joint_limits = []
    for i in range(1, 8):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{i}")
        joint_limits.append((float(model.jnt_range[joint_id][0]), float(model.jnt_range[joint_id][1])))

    hand_pos, hand_quat = body_pose("panda_hand")
    left_pos, left_quat = body_pose("panda_left_finger")
    right_pos, right_quat = body_pose("panda_right_finger")
    return {
        "capsules": tuple(capsules),
        "gripper_indices": tuple(gripper_indices),
        "joint_limits": tuple(joint_limits),
        "hand_pos": hand_pos,
        "hand_quat_wxyz": hand_quat,
        "left_finger_pos_in_hand": left_pos,
        "left_finger_quat_wxyz": left_quat,
        "right_finger_pos_in_hand": right_pos,
        "right_finger_quat_wxyz": right_quat,
        "_xml_path": str(xml_path),
    }


def capsule_radii(xml_path=str(SCENE_XML)):
    return tuple(float(c[7]) for c in load_capsule_geometry(xml_path)["capsules"])


def gripper_capsule_indices(xml_path=str(SCENE_XML)):
    """Indices of the hand and finger capsules."""
    return load_capsule_geometry(xml_path)["gripper_indices"]


def joint_limits(xml_path=str(SCENE_XML)):
    limits = load_capsule_geometry(xml_path)["joint_limits"]
    return tuple(float(l[0]) for l in limits), tuple(float(l[1]) for l in limits)


@lru_cache(maxsize=4)
def self_collision_pairs(xml_path=str(SCENE_XML)):
    """Capsule pairs checked for self-collision: bodies more than two hops apart in the kinematic tree.

    Nearby bodies are skipped because the thick capsule approximation overlaps
    there even when the actual geometry does not collide.
    """
    names = tuple(str(c[0]) for c in load_capsule_geometry(xml_path)["capsules"])
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    body_ids = {n: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)) for n in set(names)}
    body_parent = [int(p) for p in model.body_parentid]

    def ancestors(body_id):
        hops = {}
        hop = 0
        current = int(body_id)
        while current >= 0:
            if current in hops and hops[current] <= hop:
                break
            hops[current] = hop
            parent = body_parent[current]
            if parent == current:
                break
            current = parent
            hop += 1
        return hops

    ancestor_maps = {n: ancestors(b) for n, b in body_ids.items()}

    def tree_hops(a, b):
        common = set(ancestor_maps[a]).intersection(ancestor_maps[b])
        return min(ancestor_maps[a][node] + ancestor_maps[b][node] for node in common)

    pairs = []
    for idx_a, name_a in enumerate(names):
        for idx_b in range(idx_a + 1, len(names)):
            if tree_hops(name_a, names[idx_b]) <= 2:
                continue
            pairs.append((idx_a, idx_b))
    return tuple(pairs)


class PandaCapsuleFK:
    """Batched forward kinematics of the capsule segments (Panda DH parameters)."""

    def __init__(self, device="cpu", dtype=torch.float32, finger_q=0.04, xml_path=str(SCENE_XML)):
        self.device = torch.device(device)
        self.dtype = dtype
        self.default_finger_q = float(finger_q)
        self.dh = torch.tensor(
            [
                [0.0000, 0.3330, 0.0],
                [0.0000, 0.0000, -torch.pi / 2.0],
                [0.0000, 0.3160, torch.pi / 2.0],
                [0.0825, 0.0000, torch.pi / 2.0],
                [-0.0825, 0.3840, -torch.pi / 2.0],
                [0.0000, 0.0000, torch.pi / 2.0],
                [0.0880, 0.0000, torch.pi / 2.0],
            ],
            device=self.device,
            dtype=self.dtype,
        )
        geometry = load_capsule_geometry(xml_path)

        def t(value):
            return torch.tensor(value, device=self.device, dtype=self.dtype)

        self.hand_pos = t(geometry["hand_pos"])
        self.hand_quat_wxyz = t(geometry["hand_quat_wxyz"])
        self.left_finger_pos_in_hand = t(geometry["left_finger_pos_in_hand"])
        self.left_finger_quat_wxyz = t(geometry["left_finger_quat_wxyz"])
        self.right_finger_pos_in_hand = t(geometry["right_finger_pos_in_hand"])
        self.right_finger_quat_wxyz = t(geometry["right_finger_quat_wxyz"])
        capsules = geometry["capsules"]
        self.capsule_body_names = tuple(str(c[0]) for c in capsules)
        self.capsule_local_start = t([[c[1], c[2], c[3]] for c in capsules])
        self.capsule_local_end = t([[c[4], c[5], c[6]] for c in capsules])

    def _dh_transform(self, theta, a, d, alpha):
        bsz = theta.shape[0]
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)
        T = torch.zeros((bsz, 4, 4), device=theta.device, dtype=theta.dtype)
        T[:, 0, 0] = ct
        T[:, 0, 1] = -st
        T[:, 0, 2] = 0.0
        T[:, 0, 3] = a
        T[:, 1, 0] = st * ca
        T[:, 1, 1] = ct * ca
        T[:, 1, 2] = -sa
        T[:, 1, 3] = -sa * d
        T[:, 2, 0] = st * sa
        T[:, 2, 1] = ct * sa
        T[:, 2, 2] = ca
        T[:, 2, 3] = ca * d
        T[:, 3, 3] = 1.0
        return T

    def _rotation(self, quat_wxyz):
        w, x, y, z = quat_wxyz
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        return torch.tensor(
            [
                [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
            ],
            device=self.device,
            dtype=self.dtype,
        )

    def _fixed_transform(self, pos, quat_wxyz, batch_size):
        R = self._rotation(quat_wxyz)
        T = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        T[:, :3, :3] = R.unsqueeze(0).repeat(batch_size, 1, 1)
        T[:, :3, 3] = pos.view(1, 3).repeat(batch_size, 1)
        return T

    def _fixed_transform_with_batch_pos(self, pos, quat_wxyz):
        pos = pos.to(device=self.device, dtype=self.dtype)
        bsz = pos.shape[0]
        R = self._rotation(quat_wxyz)
        T = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(bsz, 1, 1)
        T[:, :3, :3] = R.unsqueeze(0).repeat(bsz, 1, 1)
        T[:, :3, 3] = pos
        return T

    def _split_arm_and_finger_q(self, q):
        if q.ndim == 1:
            q = q.unsqueeze(0)
        q = q.to(device=self.device, dtype=self.dtype)
        if q.shape[-1] < 7:
            raise ValueError("Panda FK expects q with at least 7 dims")
        arm_q = q[:, :7]
        finger_q = torch.full((q.shape[0], 2), self.default_finger_q, device=self.device, dtype=self.dtype)
        return arm_q, finger_q

    def _body_transforms(self, q, finger_q):
        bsz = q.shape[0]
        T = torch.eye(4, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(bsz, 1, 1)
        transforms = {"panda_link0": T.clone()}
        for j in range(7):
            a, d, alpha = self.dh[j]
            T = torch.bmm(T, self._dh_transform(q[:, j], a, d, alpha))
            transforms[f"panda_link{j + 1}"] = T.clone()
        T_hand = torch.bmm(T, self._fixed_transform(self.hand_pos, self.hand_quat_wxyz, bsz))
        transforms["panda_hand"] = T_hand
        left_finger_pos = self.left_finger_pos_in_hand.view(1, 3).repeat(bsz, 1).clone()
        left_finger_pos[:, 1] += finger_q[:, 0]
        right_finger_pos = self.right_finger_pos_in_hand.view(1, 3).repeat(bsz, 1).clone()
        right_finger_pos[:, 1] -= finger_q[:, 1]
        transforms["panda_left_finger"] = torch.bmm(
            T_hand, self._fixed_transform_with_batch_pos(left_finger_pos, self.left_finger_quat_wxyz)
        )
        transforms["panda_right_finger"] = torch.bmm(
            T_hand, self._fixed_transform_with_batch_pos(right_finger_pos, self.right_finger_quat_wxyz)
        )
        return transforms

    def forward_collision_segments(self, q):
        """q [B, 7] -> capsule segment start and end points [B, S, 3] in the world frame."""
        q, finger_q = self._split_arm_and_finger_q(q)
        transforms = self._body_transforms(q, finger_q)
        bsz = q.shape[0]
        seg_start = []
        seg_end = []
        for idx, name in enumerate(self.capsule_body_names):
            T_body = transforms[name]
            local_start = self.capsule_local_start[idx].view(1, 3, 1).repeat(bsz, 1, 1)
            local_end = self.capsule_local_end[idx].view(1, 3, 1).repeat(bsz, 1, 1)
            seg_start.append(torch.bmm(T_body[:, :3, :3], local_start).squeeze(-1) + T_body[:, :3, 3])
            seg_end.append(torch.bmm(T_body[:, :3, :3], local_end).squeeze(-1) + T_body[:, :3, 3])
        return torch.stack(seg_start, dim=1), torch.stack(seg_end, dim=1)
