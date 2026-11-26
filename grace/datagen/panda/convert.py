"""Conversion of the FR3 demonstrations (npz) into a robomimic hdf5 dataset.

Observations (float32): robot0_joint_pos (q), robot0_joint_vel (qd) and
goal_delta_q (q_goal - q). Action (float32): delta_q = q_des - q, the joint
target increment commanded by the expert. 10% of the demonstrations form the
validation split (random permutation with ``seed``; the paper used the
training seed 20).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ENV_NAME = "PandaTrackingOffline"
ENV_TYPE = 2  # robomimic EnvType.GYM_TYPE (the environment is never instantiated by robomimic)


def episode_observations(episode):
    q = np.asarray(episode["observed_q"], dtype=np.float32)
    horizon = q.shape[0]
    q_goal = np.repeat(np.asarray(episode["q_goal"], dtype=np.float32)[None, :], horizon, axis=0)
    return {
        "robot0_joint_pos": q,
        "robot0_joint_vel": np.asarray(episode["observed_qd"], dtype=np.float32),
        "goal_delta_q": q_goal - q,
    }


def episode_actions(episode):
    return np.asarray(episode["desired_q"], dtype=np.float32) - np.asarray(episode["observed_q"], dtype=np.float32)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.item()) if value.shape == () else [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def convert(input_path, output_path, validation_ratio=0.1, seed=20):
    with np.load(input_path, allow_pickle=True) as payload:
        metadata = {k: payload[k] for k in payload.files if k != "episodes"}
        episodes = [item.item() if hasattr(item, "item") else item for item in payload["episodes"]]
    episodes = [
        ep
        for ep in episodes
        if bool(ep["goal_reached"]) and not (bool(ep["had_self_collision"]) or bool(ep["had_floor_collision"]))
    ]
    if not episodes:
        raise RuntimeError("no accepted demonstrations in {}".format(input_path))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(episodes))
    num_valid = 0 if validation_ratio <= 0.0 or len(episodes) <= 1 else min(max(1, int(round(len(episodes) * validation_ratio))), len(episodes) - 1)
    valid_indices = set(int(i) for i in perm[:num_valid])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env_args = {
        "env_name": ENV_NAME,
        "type": ENV_TYPE,
        "env_kwargs": {
            "control_mode": str(metadata["control_mode"].item()),
            "action_source": "delta_q",
            "source_dataset": "tracking_dataset_npz",
            "source_metadata": {k: _jsonable(v) for k, v in metadata.items()},
        },
    }
    train_keys, valid_keys = [], []
    total = 0
    with h5py.File(output_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        for idx, episode in enumerate(episodes):
            key = f"demo_{idx}"
            demo = data.create_group(key)
            obs_group = demo.create_group("obs")
            for obs_key, value in episode_observations(episode).items():
                obs_group.create_dataset(obs_key, data=value, compression="gzip")
            actions = episode_actions(episode)
            horizon = int(actions.shape[0])
            demo.create_dataset("actions", data=actions, compression="gzip")
            demo.create_dataset("rewards", data=np.zeros((horizon, 1), dtype=np.float32), compression="gzip")
            dones = np.zeros((horizon, 1), dtype=np.float32)
            dones[-1, 0] = 1.0
            demo.create_dataset("dones", data=dones, compression="gzip")
            demo.attrs["num_samples"] = horizon
            total += horizon
            (valid_keys if idx in valid_indices else train_keys).append(key)
        data.attrs["total"] = total
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.asarray(train_keys, dtype="S"))
        if valid_keys:
            mask.create_dataset("valid", data=np.asarray(valid_keys, dtype="S"))
    return {"num_demos": len(episodes), "num_valid": len(valid_keys), "num_samples": total, "output": str(output_path)}
