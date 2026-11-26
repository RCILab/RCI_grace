"""Parallel collection of the FR3 demonstration dataset (paper Table I: 20,000 obstacle-free demonstrations).

Attempt k draws its start/goal pair from ``np.random.SeedSequence([seed, 0,
k])`` and attempts are dispatched in order, so the dataset holds the first
``num_episodes`` accepted attempts whatever the number of workers. Running the
same command again resumes an interrupted collection: the attempts already in
the snapshot are skipped, so the result equals that of an uninterrupted run.
Episodes are stored in attempt order. The output is an npz file with the list
of episodes and the configuration (converted to robomimic hdf5 by ``convert``).
"""

from __future__ import annotations

import concurrent.futures as futures
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from grace.datagen.panda.expert import generate_episode


def config_repr(config):
    return str(asdict(config))


def _is_accepted(episode):
    return bool(episode["goal_reached"]) and not (bool(episode["had_self_collision"]) or bool(episode["had_floor_collision"]))


def _attempt(seed_seq, config, attempt_index):
    try:
        episode = generate_episode(np.random.default_rng(seed_seq), config)
    except RuntimeError as exc:
        if str(exc).startswith("Failed to sample"):
            return None
        raise
    if not _is_accepted(episode):
        return None
    episode["attempt_index"] = np.array(attempt_index, dtype=int)
    return episode


def save_snapshot(path, episodes, config, requested):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = sorted(episodes, key=lambda ep: int(ep["attempt_index"]))
    payload = {
        "control_mode": np.array("torque", dtype=object),
        "config": np.array(config_repr(config), dtype=object),
        "num_episodes": np.array(len(episodes), dtype=int),
        "requested_num_episodes": np.array(requested, dtype=int),
        "collection_complete": np.array(len(episodes) >= requested, dtype=bool),
        "episodes": np.array(episodes, dtype=object),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(path)


def load_snapshot(path, config):
    path = Path(path)
    if not path.exists():
        return []
    with np.load(path, allow_pickle=True) as payload:
        if str(payload["config"].item()) != config_repr(config):
            raise RuntimeError(f"{path} was collected with a different config; use another output path")
        return [item.item() if hasattr(item, "item") else item for item in payload["episodes"]]


def generate_dataset(output, num_episodes, config, num_workers=8, save_every=100):
    episodes = load_snapshot(output, config)
    if len(episodes) >= num_episodes:
        print(f"{output} already holds {len(episodes)} episodes", flush=True)
        return output
    collected = {int(ep["attempt_index"]) for ep in episodes}
    if len(collected) != len(episodes):
        raise RuntimeError(f"{output} holds several episodes of the same attempt; use another output path")
    if episodes:
        print(f"resuming at {len(episodes)}/{num_episodes} episodes", flush=True)
    seed = 0 if config.seed is None else int(config.seed)
    attempt = 0
    last_saved = len(episodes)
    with futures.ProcessPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        inflight = {}

        def fill():
            nonlocal attempt
            while len(inflight) < max(1, int(num_workers)) and len(episodes) + len(inflight) < num_episodes:
                if attempt not in collected:  # attempts held by a resumed snapshot are skipped
                    seed_seq = np.random.SeedSequence([seed, 0, attempt])  # the 0 keeps the streams of the paper dataset
                    inflight[executor.submit(_attempt, seed_seq, config, attempt)] = attempt
                attempt += 1

        fill()
        try:
            while inflight and len(episodes) < num_episodes:
                done, _ = futures.wait(inflight, return_when=futures.FIRST_COMPLETED)
                for fut in done:
                    inflight.pop(fut)
                    episode = fut.result()
                    if episode is None:
                        continue
                    episodes.append(episode)
                    if len(episodes) - last_saved >= save_every or len(episodes) == num_episodes:
                        save_snapshot(output, episodes, config, num_episodes)
                        last_saved = len(episodes)
                    print(f"episode {len(episodes)}/{num_episodes} (attempts {attempt})", flush=True)
                fill()
        finally:
            if len(episodes) != last_saved:
                save_snapshot(output, episodes, config, num_episodes)
    print(json.dumps({"output": str(output), "num_episodes": len(episodes), "attempts": attempt}), flush=True)
    return output
