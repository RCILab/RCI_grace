"""Synchronous evaluation in the planar point-mass task (paper Table II).

Simulated time does not advance during planning. Each episode samples a new
start/goal pair, then alternates between planning (every H_exec steps) and
stepping the environment, until success, collision or the step budget.
"""

import json
import os
import time

import numpy as np
import torch

from grace.envs.pointmass2d import make_env
from grace.envs.pointmass2d.geometry import path_length
from grace.eval.metrics import summarize_episodes
from grace.guidance import AffineNormalizer, GraceConfig, GraceSampler
from grace.policy import ObservationHistory, load_prior, make_closed_loop_policy
from grace.tasks.pointmass2d import PointMassGuidanceTask


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_episode(policy, env, history, max_steps, device):
    planner = policy.policy
    sampler = planner.sampler
    policy.start_episode()
    obs = history.reset(env.reset())
    positions = [env.position.copy()]
    actions = []
    planning_times_ms, guidance_plan_times_ms = [], []
    info = {}
    success = False
    for _ in range(max_steps):
        will_plan = planner.will_plan
        _sync(device)
        tic = time.perf_counter()
        action = policy(ob=obs)
        _sync(device)
        if will_plan:
            planning_times_ms.append((time.perf_counter() - tic) * 1000.0)
            if sampler.guided:
                guidance_plan_times_ms.append(float(np.sum(sampler.last_guidance_times_ms)))
        next_obs, _, done, info = env.step(action)
        positions.append(env.position.copy())
        actions.append(np.asarray(action, dtype=np.float32).copy())
        success = env.last_success
        if done or success:
            break
        obs = history.push(next_obs)

    if success:
        failure = None
    elif info.get("collision", False):
        failure = "collision"
    else:
        failure = "timeout"
    return {
        "success": bool(success),
        "failure": failure,
        "steps": len(actions),
        "path_length": path_length(positions),
        "goal_distance_final": float(info.get("goal_distance", np.nan)),
        "planning_times_ms": planning_times_ms,
        "guidance_plan_times_ms": guidance_plan_times_ms,
        "positions": np.asarray(positions, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1, 2),
    }


def evaluate(config):
    """``config``: dict as in configs/eval/pointmass2d_*.yaml."""
    if torch.cuda.is_available() and config.get("device", "cuda") == "cuda":
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    rollout_policy, ckpt_dict = load_prior(config["checkpoint"], device)
    env = make_env(config["env"], ckpt_dict["env_metadata"]["env_kwargs"])
    algo = rollout_policy.policy

    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.seed(seed)

    guidance = config.get("guidance")
    if guidance:
        cfg = GraceConfig(**guidance)
        normalizer = AffineNormalizer.from_rollout_policy(rollout_policy, device)
        task = PointMassGuidanceTask(env, w_obs=config["cost"]["w_obs"], normalizer=normalizer, device=device)
        sampler = GraceSampler(algo, cfg=cfg, task=task, normalizer=normalizer)
    else:
        sampler = GraceSampler(algo)
    policy = make_closed_loop_policy(rollout_policy, sampler)
    history = ObservationHistory(algo.global_config.train.frame_stack)

    episodes = []
    for i in range(int(config["num_episodes"])):
        ep = run_episode(policy, env, history, int(config["max_steps"]), device)
        episodes.append(ep)
        print(
            "episode {:3d}: {:8s} steps={:4d} path_length={:.3f}".format(
                i, "success" if ep["success"] else ep["failure"], ep["steps"], ep["path_length"]
            ),
            flush=True,
        )
    summary = summarize_episodes(episodes)
    print(json.dumps(summary, indent=2))

    output_dir = config.get("output_dir")
    if output_dir:
        save_results(output_dir, config, summary, episodes)
    return summary, episodes


def save_results(output_dir, config, summary, episodes):
    os.makedirs(output_dir, exist_ok=True)
    scalar_keys = ("success", "failure", "steps", "path_length", "goal_distance_final")
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "config": config,
                "summary": summary,
                "episodes": [{k: ep[k] for k in scalar_keys} for ep in episodes],
            },
            f,
            indent=2,
        )
    np.savez_compressed(
        os.path.join(output_dir, "trajectories.npz"),
        positions=np.array([ep["positions"] for ep in episodes], dtype=object),
        actions=np.array([ep["actions"] for ep in episodes], dtype=object),
    )
