"""Writing demonstrations in the robomimic hdf5 format (data/demo_i, mask/train, mask/valid)."""

import json
import os

import h5py
import numpy as np


def _sorted_demo_keys(data_group):
    return sorted(list(data_group.keys()), key=lambda key: int(key.split("_")[1]))


def split_train_valid(demo_keys, val_ratio=0.1, seed=0):
    """Random train/validation split of the demonstrations (validation ratio ``val_ratio``)."""
    demo_keys = list(demo_keys)
    if val_ratio is None or val_ratio <= 0.0 or len(demo_keys) <= 1:
        return demo_keys, []
    rng = np.random.default_rng(seed)
    num_valid = min(max(1, int(round(len(demo_keys) * val_ratio))), len(demo_keys) - 1)
    valid_indices = set(int(i) for i in rng.permutation(len(demo_keys))[:num_valid])
    train_keys = [k for i, k in enumerate(demo_keys) if i not in valid_indices]
    valid_keys = [k for i, k in enumerate(demo_keys) if i in valid_indices]
    return train_keys, valid_keys


def write_masks(h5_file, train_keys, valid_keys):
    mask_group = h5_file.require_group("mask")
    for key in list(mask_group.keys()):
        del mask_group[key]
    mask_group.create_dataset("train", data=np.asarray(train_keys, dtype="S"))
    if valid_keys:
        mask_group.create_dataset("valid", data=np.asarray(valid_keys, dtype="S"))


def _data_group(h5_file, env_metadata):
    data_grp = h5_file.require_group("data")
    data_grp.attrs["env_args"] = json.dumps(env_metadata)
    if "total" not in data_grp.attrs:
        data_grp.attrs["total"] = 0
    return data_grp


def resume_demos(path, env_metadata, marker):
    """Prepares a shard for resuming and returns the attributes of its demonstrations, in order.

    ``append_demo`` writes the ``attrs`` of a demonstration last, so the last
    demonstration lacks ``marker`` if its write was interrupted; it is deleted.
    """
    if not os.path.exists(path):
        return []
    try:
        f = h5py.File(path, "a")
    except OSError as exc:
        raise RuntimeError("Cannot open {} (interrupted while writing?); delete it to regenerate this shard".format(path)) from exc
    with f:
        if "data" not in f:
            return []
        data_grp = f["data"]
        keys = _sorted_demo_keys(data_grp)
        if keys and data_grp.attrs.get("env_args") != json.dumps(env_metadata):
            raise RuntimeError("{} was generated with a different configuration; delete it or use another output path".format(path))
        if keys and marker not in data_grp[keys[-1]].attrs:
            del data_grp[keys.pop()]
        if any(marker not in data_grp[key].attrs for key in keys):
            raise RuntimeError("{} holds demonstrations without the attribute {!r}; delete it to regenerate this shard".format(path, marker))
        data_grp.attrs["total"] = sum(int(data_grp[key].attrs["num_samples"]) for key in keys)
        return [dict(data_grp[key].attrs) for key in keys]


def append_demo(path, demo, env_metadata, attrs=None):
    """Appends one demonstration: obs/next_obs dicts, actions, rewards, dones (+ start/goal attrs).

    ``attrs`` are written last, so a demonstration whose write was interrupted
    lacks them (see ``resume_demos``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "a") as f:
        data_grp = _data_group(f, env_metadata)
        ep_grp = data_grp.create_group("demo_{}".format(len(data_grp.keys())))
        num_samples = int(demo["actions"].shape[0])
        ep_grp.attrs["num_samples"] = num_samples
        for key in ("start", "goal"):
            if key in demo:
                ep_grp.attrs[key] = demo[key]
        ep_grp.create_dataset("actions", data=demo["actions"].astype(np.float32), compression="gzip")
        ep_grp.create_dataset("rewards", data=demo["rewards"].astype(np.float32), compression="gzip")
        ep_grp.create_dataset("dones", data=demo["dones"].astype(np.bool_), compression="gzip")
        for group in ("obs", "next_obs"):
            obs_grp = ep_grp.create_group(group)
            for key, value in demo[group].items():
                obs_grp.create_dataset(key, data=np.asarray(value, dtype=np.float32), compression="gzip")
        data_grp.attrs["total"] = int(data_grp.attrs["total"]) + num_samples
        for key, value in (attrs or {}).items():
            ep_grp.attrs[key] = value


def merge_shards(output_path, shard_paths, env_metadata, val_ratio=0.1, seed=0):
    """Concatenates worker shards (in order) and writes the train/valid masks."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    total_samples = 0
    total_demos = 0
    with h5py.File(output_path, "w") as out:
        out_data = _data_group(out, env_metadata)
        for shard_path in shard_paths:
            with h5py.File(shard_path, "r") as shard:
                for key in _sorted_demo_keys(shard["data"]):
                    shard.copy(shard["data"][key], out_data, name="demo_{}".format(total_demos))
                    total_samples += int(shard["data"][key].attrs["num_samples"])
                    total_demos += 1
        out_data.attrs["total"] = total_samples
        train_keys, valid_keys = split_train_valid(_sorted_demo_keys(out_data), val_ratio=val_ratio, seed=seed)
        write_masks(out, train_keys, valid_keys)
    return {"num_demos": total_demos, "num_samples": total_samples, "num_valid": len(valid_keys)}
