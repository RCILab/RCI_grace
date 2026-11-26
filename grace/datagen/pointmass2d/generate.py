"""Generation of the point-mass demonstration dataset (paper Table I: 10,000 demonstrations, base map).

Each worker samples start/goal pairs, plans a path with OMPL, converts it into
velocity commands and replays them in the environment; only demonstrations
that reach the goal without collision are stored. Worker ``w`` uses the seed
``seed + 10000 * w``. Note that OMPL's internal random state is not seeded, so
the planned paths (unlike the start/goal pairs) differ between runs.

An interrupted run is resumed by running the same command again: each worker
keeps the complete demonstrations of its shard and replays the start/goal draws
of the attempts behind them, so the start/goal pairs are those of an
uninterrupted run.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from robomimic.envs.env_base import EnvType

from grace.datagen import hdf5
from grace.datagen.pointmass2d.expert import PointMassExpert
from grace.envs.pointmass2d import ENV_ID_TRAIN, PointMassConfig, PointMassEnv, build_observation_dict


def env_metadata(config):
    return {"env_name": ENV_ID_TRAIN, "type": EnvType.GYM_TYPE, "env_kwargs": config.env_kwargs()}


def rollout_demo(env, start, goal, actions):
    """Replays the commands from ``start``; returns the demonstration if it reaches the goal."""
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    zero = np.zeros(2, dtype=np.float32)
    env.set_state({"position": start, "velocity": zero, "goal": goal, "step": 0})
    current = build_observation_dict(start, zero, goal)
    obs = {key: [] for key in current}
    next_obs = {key: [] for key in current}
    executed, rewards, dones = [], [], []
    info = {"success": False}
    for action in actions:
        action = np.asarray(action, dtype=np.float32)
        _, reward, done, info = env.step(action)
        nxt = build_observation_dict(env.position, env.velocity, goal)
        for key in obs:
            obs[key].append(current[key])
            next_obs[key].append(nxt[key])
        executed.append(action)
        rewards.append(np.float32(reward))
        dones.append(bool(done))
        current = nxt
        if done:
            break
    if not bool(info["success"]):
        return None
    return {
        "obs": {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()},
        "next_obs": {k: np.asarray(v, dtype=np.float32) for k, v in next_obs.items()},
        "actions": np.asarray(executed, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "start": start,
        "goal": goal,
    }


def generate_worker(worker_id, num_demos, config_kwargs, seed, max_attempts_per_demo, shard_path):
    config = PointMassConfig(**config_kwargs)
    env = PointMassEnv(config=config, use_extra_obstacles=False)
    expert = PointMassExpert(env.map, config, rng=np.random.RandomState(seed))
    metadata = env_metadata(config)

    # Resume: keep the complete demonstrations of the shard and replay the
    # start/goal draws of the attempts that produced them, so the worker
    # continues the random stream of an uninterrupted run.
    stored = hdf5.resume_demos(shard_path, metadata, marker="attempt")
    generated = len(stored)
    if generated > num_demos:
        raise RuntimeError("Shard {} already holds {} > {} demos".format(shard_path, generated, num_demos))
    attempts = int(stored[-1]["attempt"]) if stored else 0
    for _ in range(attempts):
        expert.sample_start_goal()
    if 0 < generated < num_demos:
        print("worker {}: resuming at {}/{} demos (attempt {})".format(worker_id, generated, num_demos, attempts), flush=True)
    max_attempts = attempts + max(num_demos - generated, 0) * max(max_attempts_per_demo, 1)
    while generated < num_demos and attempts < max_attempts:
        attempts += 1
        start, goal = expert.sample_start_goal()
        path = expert.plan(start, goal)
        if path is None:
            continue
        actions = expert.path_to_actions(path)
        if actions is None:
            continue
        demo = rollout_demo(env, start, goal, actions)
        if demo is None:
            continue
        hdf5.append_demo(shard_path, demo, metadata, attrs={"attempt": attempts})
        generated += 1
        if generated % 50 == 0 or generated == num_demos:
            print("worker {}: {}/{} demos".format(worker_id, generated, num_demos), flush=True)
    if generated < num_demos:
        raise RuntimeError("worker {} generated only {}/{} demos".format(worker_id, generated, num_demos))
    return shard_path


def generate_dataset(output, num_demos=10000, num_workers=16, seed=0, val_ratio=0.1, max_attempts_per_demo=100, config=None):
    config = config or PointMassConfig()
    num_workers = max(1, min(int(num_workers), int(num_demos)))
    counts = [num_demos // num_workers + (1 if w < num_demos % num_workers else 0) for w in range(num_workers)]
    shard_dir = "{}__shards_s{}_w{}_n{}".format(os.path.splitext(os.path.abspath(output))[0], seed, num_workers, num_demos)
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = [os.path.join(shard_dir, "worker_{:03d}.hdf5".format(w)) for w in range(num_workers)]
    jobs = [
        (w, counts[w], config.env_kwargs(), seed + 10000 * w, max_attempts_per_demo, shard_paths[w])
        for w in range(num_workers)
    ]
    if num_workers == 1:
        generate_worker(*jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for future in as_completed([executor.submit(generate_worker, *job) for job in jobs]):
                future.result()
    return hdf5.merge_shards(output, shard_paths, env_metadata(config), val_ratio=val_ratio, seed=seed)
