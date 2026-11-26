"""Synchronous evaluation of the FR3 goal-reaching task (paper Table IV).

Simulated time does not advance during planning. Per trial:

  1. a random start/goal pair and six unseen obstacles are drawn
     (``panda_scenario``), the arm is reset to the start;
  2. every control step the state is checked, in this order, for
     self-collision, floor contact, joint-limit violation, obstacle contact,
     goal reached (max_j |q_j - q_goal_j| <= goal tolerance) and the step budget;
  3. when no planned action is left, a new plan is sampled from the
     observation; its first H_exec delta_q actions are executed one per control
     step. Each action moves the latched joint target (anchored at the measured
     q when the plan was made) and is tracked by the torque controller.

Before executing the first plan of a trial, one control step holds the
current configuration (as in the paper runs, where it was used to display the
first plan); after the trial ends one more hold step is simulated.
"""

from __future__ import annotations

import json
import os
import time

import mujoco
import numpy as np
import torch

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils
import robomimic.utils.tensor_utils as TensorUtils

from grace.envs.panda.config import SCENE_XML, PandaConfig
from grace.envs.panda.controllers import clip_joint_velocity
from grace.envs.panda.kinematics import PandaPinocchio
from grace.envs.panda.sim import ARM_JOINTS, FINGER_JOINTS, PandaEvalSim
from grace.eval.metrics import summarize_episodes
from grace.eval.panda_scenario import ScenarioConfig, joint_limit_violation, make_trial
from grace.guidance import AffineNormalizer, GraceConfig, GraceSampler
from grace.policy import load_prior
from grace.tasks.panda import PandaGuidanceTask

OBS_KEYS = ("goal_delta_q", "robot0_joint_pos", "robot0_joint_vel")


class ObstacleChecker:
    """Exact robot-obstacle contact test with MuJoCo.

    A second copy of the scene holds the trial's obstacles in the obstacle
    slots of scene.xml (shape_obstacle_<shape>_<k>). A configuration collides
    if MuJoCo reports a contact between an arm geom and an obstacle, or if their
    signed distance is below -``penetration_tolerance``.
    """

    SHAPES = ("sphere", "box", "tri_prism", "diamond_prism")

    def __init__(self, config, xml_path=SCENE_XML, penetration_tolerance=1e-6):
        self.config = config
        self.tolerance = float(penetration_tolerance)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.robot_body_ids = {
            b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("panda_")
        }
        self.robot_geom_ids = np.array(
            [
                g
                for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) in self.robot_body_ids and int(m.geom_contype[g]) != 0 and int(m.geom_conaffinity[g]) != 0
            ],
            dtype=int,
        )
        self.obstacle_geom_ids = np.zeros(0, dtype=int)
        joint_qpos = lambda name: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
        self.arm_qpos_idx = np.array([joint_qpos(n) for n in ARM_JOINTS], dtype=int)
        self.finger_qpos_idx = np.array([joint_qpos(n) for n in FINGER_JOINTS], dtype=int)

    def _geom_name(self, geom_id):
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        return "" if name is None else str(name)

    def _is_obstacle_geom(self, geom_id):
        name = self._geom_name(geom_id)
        return name.startswith("shape_obstacle_") or name.startswith("obstacle_marker_")

    def _hide_all(self):
        m, d = self.model, self.data
        slots = [("obstacle_marker_{:02d}".format(i), -30.0 - i) for i in range(64)]
        groups = [slots] + [[("shape_obstacle_{}_{:02d}".format(s, i), -35.0 - i) for i in range(64)] for s in self.SHAPES]
        for group in groups:
            for body_name, z in group:
                body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
                geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, body_name + "_geom")
                if body_id < 0 and geom_id < 0:
                    break
                if body_id >= 0 and int(m.body_mocapid[body_id]) >= 0:
                    d.mocap_pos[m.body_mocapid[body_id]] = np.asarray([0.0, 0.0, z], dtype=np.float64)
                    d.mocap_quat[m.body_mocapid[body_id]] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                if geom_id >= 0:
                    m.geom_contype[geom_id] = 0
                    m.geom_conaffinity[geom_id] = 0
                    m.geom_size[geom_id, :] = 1e-4

    def set_obstacles(self, specs):
        m, d = self.model, self.data
        self._hide_all()
        active = []
        counters = {shape: 0 for shape in self.SHAPES}
        for spec in specs:
            shape = str(spec.get("shape", "sphere"))
            idx = counters.get(shape, 0)
            counters[shape] = idx + 1
            body_name = "shape_obstacle_{}_{:02d}".format(shape, idx)
            body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, body_name + "_geom")
            if body_id < 0 or geom_id < 0 or int(m.body_mocapid[body_id]) < 0:
                raise RuntimeError("scene.xml has no free slot for obstacle {} ({})".format(idx, shape))
            quat = np.asarray(spec.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(4)
            quat_norm = float(np.linalg.norm(quat))
            quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if quat_norm <= 1e-8 else quat / quat_norm
            half_extents = np.asarray(spec["half_extents"], dtype=np.float64).reshape(3)
            d.mocap_pos[m.body_mocapid[body_id]] = np.asarray(spec["center"], dtype=np.float64).reshape(3)
            d.mocap_quat[m.body_mocapid[body_id]] = quat
            if shape == "sphere":
                m.geom_size[geom_id, 0] = max(float(spec.get("radius", half_extents[0])), 1e-4)
            else:
                m.geom_size[geom_id, :] = np.maximum(half_extents, 1e-4)
            m.geom_contype[geom_id] = 1
            m.geom_conaffinity[geom_id] = 1
            active.append(int(geom_id))
        self.obstacle_geom_ids = np.asarray(active, dtype=int)
        mujoco.mj_forward(m, d)

    def collides(self, q_arm):
        m, d = self.model, self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[self.arm_qpos_idx] = np.asarray(q_arm, dtype=float).reshape(-1)[:7]
        d.qpos[self.finger_qpos_idx] = float(self.config.finger_q)
        mujoco.mj_forward(m, d)
        for idx in range(int(d.ncon)):
            g1, g2 = int(d.contact[idx].geom1), int(d.contact[idx].geom2)
            r1 = int(m.geom_bodyid[g1]) in self.robot_body_ids
            r2 = int(m.geom_bodyid[g2]) in self.robot_body_ids
            if (r1 and self._is_obstacle_geom(g2)) or (r2 and self._is_obstacle_geom(g1)):
                return True
        fromto = np.zeros(6, dtype=np.float64)
        for robot_geom in self.robot_geom_ids:
            for obstacle_geom in self.obstacle_geom_ids:
                fromto.fill(0.0)
                try:
                    distance = float(mujoco.mj_geomDistance(m, d, int(robot_geom), int(obstacle_geom), 0.0, fromto))
                except Exception:
                    continue
                if np.isfinite(distance) and distance < -self.tolerance and distance <= 1e100:
                    return True
        return False


def unnormalize_actions(rollout_policy, actions):
    """robomimic action un-normalization of an [T, m] sequence (as RolloutPolicy does for one action)."""
    stats = rollout_policy.action_normalization_stats
    action_keys = rollout_policy.policy.global_config.train.action_keys
    shapes = {k: stats[k]["offset"].shape[1:] for k in stats}
    action_dict = PyUtils.vector_to_action_dict(actions, action_shapes=shapes, action_keys=action_keys)
    action_dict = ObsUtils.unnormalize_dict(action_dict, normalization_stats=stats)
    return PyUtils.action_dict_to_vector(action_dict, action_keys=action_keys)


def policy_observation(obs, q_goal):
    q = np.asarray(obs["q"], dtype=np.float32)
    return {
        "robot0_joint_pos": q,
        "robot0_joint_vel": np.asarray(obs["qd"], dtype=np.float32),
        "goal_delta_q": np.asarray(q_goal, dtype=np.float32) - q,
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _path_length(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


class DiffusionPlanner:
    """Samples a plan (H_exec delta_q actions, raw units) from the prior, guided by GRACE if configured."""

    def __init__(self, rollout_policy, sampler, task=None):
        self.rollout_policy = rollout_policy
        self.sampler = sampler
        self.task = task
        self.H_exec = sampler.H_exec
        self.last_guidance_ms = None

    def reset(self, trial):
        if self.task is not None:
            self.task.set_obstacles(trial.obstacles)
        self.rollout_policy.start_episode()

    def __call__(self, obs, q_goal):
        ob = self.rollout_policy._prepare_observation({k: v[None] for k, v in policy_observation(obs, q_goal).items()})
        if self.task is not None:
            # GRACE rolls the candidates out from the measured q (latched joint target).
            self.task.set_latched_q_des(obs["q"])
        U = self.sampler.sample(ob)
        self.last_guidance_ms = float(np.sum(self.sampler.last_guidance_times_ms)) if self.sampler.guided else None
        return np.asarray(unnormalize_actions(self.rollout_policy, TensorUtils.to_numpy(U)[0]), dtype=np.float32)


def run_trial(trial, sim, checker, planner, cfg, run_cfg, device):
    """Runs one trial; ``planner(obs, q_goal)`` returns the next H_exec actions."""
    action_clip = np.asarray(run_cfg["action_clip"], dtype=float)
    max_steps = int(run_cfg["max_steps"])
    goal_tol = float(run_cfg["goal_joint_tolerance"])
    H_exec = planner.H_exec

    checker.set_obstacles(trial.obstacles)
    planner.reset(trial)
    sim.reset(trial.q_start)

    plan = []
    plan_q0 = None
    q_des_latched = prev_q_des = None
    first_plan_pending = True
    step = 0
    qs, ees = [], []
    planning_times_ms, guidance_times_ms = [], []
    while True:
        obs = sim.observation()
        qs.append(obs["q"])
        ees.append(obs["ee_pos"])
        self_c, floor_c = sim.collision_flags()
        if self_c:
            reason = "self_collision"
        elif floor_c:
            reason = "floor_collision"
        elif joint_limit_violation(obs["q"], run_cfg["joint_limit_tolerance"]) > 0.0:
            reason = "joint_limit"
        elif checker.collides(obs["q"]):
            reason = "obstacle_collision"
        elif float(np.max(np.abs(np.asarray(obs["q"], dtype=float) - trial.q_goal))) <= goal_tol:
            reason = "goal_reached"
        elif step >= max_steps:
            reason = "timeout"
        else:
            reason = None
        if reason is not None:
            break

        if not plan:
            # Plan from the current observation (simulated time is paused).
            _sync(device)
            tic = time.perf_counter()
            plan_q0 = np.asarray(obs["q"], dtype=float).copy()
            actions = np.clip(np.asarray(planner(obs, trial.q_goal), dtype=float), -action_clip, action_clip)
            plan = list(actions[:H_exec])
            _sync(device)
            planning_times_ms.append((time.perf_counter() - tic) * 1000.0)
            if planner.last_guidance_ms is not None:
                guidance_times_ms.append(planner.last_guidance_ms)
            if first_plan_pending:
                first_plan_pending = False
                sim.step(sim.gravity_hold_torque())
                continue

        # Execute the next delta_q action (latched joint target).
        if len(plan) == H_exec:
            q_des_latched = plan_q0.copy()
            prev_q_des = plan_q0.copy()
        action = np.clip(np.asarray(plan.pop(0), dtype=float).copy(), -action_clip, action_clip)
        q_des = q_des_latched + action
        q_des_latched = q_des.copy()
        qd_des = clip_joint_velocity((q_des - prev_q_des) / float(cfg.control_dt), np.asarray(cfg.max_joint_velocity, dtype=float))
        prev_q_des = q_des.copy()
        sim.step(sim.torque_for_command(q_des, qd_des))
        step += 1

    sim.step(sim.gravity_hold_torque())
    qs.append(qs[-1])
    ees.append(ees[-1])
    success = reason == "goal_reached"
    return {
        "seed": int(trial.seed),
        "success": bool(success),
        "failure": None if success else reason,
        "steps": int(step),
        "joint_path_length": _path_length(qs),
        "ee_path_length": _path_length(ees),
        "planning_times_ms": planning_times_ms,
        "guidance_plan_times_ms": guidance_times_ms,
        "q_start": trial.q_start,
        "q_goal": trial.q_goal,
        "obstacles": trial.obstacles,
        "q_trajectory": np.asarray(qs, dtype=np.float32),
    }


def evaluate(config):
    """``config``: dict as in configs/eval/panda_*.yaml."""
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.get("device", "cuda") == "cuda" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    rollout_policy, ckpt_dict = load_prior(config["checkpoint"], device)
    env_kwargs = ckpt_dict["env_metadata"]["env_kwargs"]
    if env_kwargs.get("action_source") != "delta_q" or env_kwargs.get("control_mode") != "torque":
        raise ValueError("expected a delta_q / torque checkpoint")
    cfg = PandaConfig.from_saved(env_kwargs["source_metadata"]["config"])
    algo = rollout_policy.policy

    seed = config.get("torch_seed")
    if seed is not None:  # the paper runs did not seed torch
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    guidance = config.get("guidance")
    task = None
    if guidance:
        normalizer = AffineNormalizer.from_rollout_policy(rollout_policy, device)
        c = config["cost"]
        task = PandaGuidanceTask(
            cfg,
            normalizer,
            device,
            w_obs=c["w_obs"],
            w_self=c["w_self"],
            w_floor=c["w_floor"],
            w_jl=c["w_jl"],
            floor_height=c["floor_height"],
            joint_limit_margin=c["joint_limit_margin"],
            action_clip=config["action_clip"],
        )
        sampler = GraceSampler(algo, cfg=GraceConfig(**guidance), task=task, normalizer=normalizer)
    else:
        sampler = GraceSampler(algo)
    planner = DiffusionPlanner(rollout_policy, sampler, task)

    sim = PandaEvalSim(cfg)
    checker = ObstacleChecker(cfg)
    kin = PandaPinocchio(ee_frame=cfg.ee_frame)
    scenario = ScenarioConfig()
    episodes = []
    for i in range(int(config["num_trials"])):
        trial = make_trial(cfg, scenario, int(config["seed"]), i, kin=kin)
        ep = run_trial(trial, sim, checker, planner, cfg, config, device)
        episodes.append(ep)
        print(
            "trial {:3d} (seed {:3d}): {:20s} steps={:5d}".format(
                i, ep["seed"], "success" if ep["success"] else ep["failure"], ep["steps"]
            ),
            flush=True,
        )
    summary = summarize_episodes([{**ep, "path_length": ep["joint_path_length"]} for ep in episodes])
    print(json.dumps(summary, indent=2))
    output_dir = config.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        keys = ("seed", "success", "failure", "steps", "joint_path_length", "ee_path_length", "planning_times_ms", "guidance_plan_times_ms")
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "config": config,
                    "summary": summary,
                    "trials": [
                        {**{k: ep[k] for k in keys}, "q_start": ep["q_start"].tolist(), "q_goal": ep["q_goal"].tolist(), "obstacles": ep["obstacles"]}
                        for ep in episodes
                    ],
                },
                f,
                indent=2,
            )
        np.savez_compressed(
            os.path.join(output_dir, "trajectories.npz"),
            q=np.array([ep["q_trajectory"] for ep in episodes], dtype=object),
        )
    return summary, episodes
