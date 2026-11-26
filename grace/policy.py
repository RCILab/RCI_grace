"""Closed-loop execution of a trained diffusion prior, with or without GRACE guidance.

The prior is loaded with robomimic (``policy_from_checkpoint``), which also
provides the observation / action normalization of the checkpoint through
``RolloutPolicy``. ``RecedingHorizonPlanner`` takes the place of the algo's
``get_action``: every H_exec control steps it samples a new plan with a
``GraceSampler`` and then returns its actions one by one (Alg. 1, line 16).
"""

from collections import deque

import numpy as np

import robomimic.utils.file_utils as FileUtils
from robomimic.algo import RolloutPolicy


def load_prior(ckpt_path, device):
    """Returns (robomimic RolloutPolicy, checkpoint dict)."""
    return FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)


class RecedingHorizonPlanner:
    """Algo adapter used by ``RolloutPolicy``: plan, execute H_exec actions, replan."""

    def __init__(self, algo, sampler):
        self.algo = algo
        self.sampler = sampler
        self.action_queue = deque()

    # attributes RolloutPolicy reads from the algo
    @property
    def device(self):
        return self.algo.device

    @property
    def global_config(self):
        return self.algo.global_config

    def set_eval(self):
        self.algo.set_eval()

    def reset(self):
        self.algo.reset()
        self.action_queue.clear()

    @property
    def will_plan(self):
        return len(self.action_queue) == 0

    def get_action(self, obs_dict, goal_dict=None):
        if len(self.action_queue) == 0:
            self.action_queue.extend(self.sampler.sample(obs_dict)[0])
        return self.action_queue.popleft().unsqueeze(0)


def make_closed_loop_policy(rollout_policy, sampler):
    """``RolloutPolicy`` (same pre/post-processing as robomimic) around a GraceSampler."""
    return RolloutPolicy(
        RecedingHorizonPlanner(rollout_policy.policy, sampler),
        obs_normalization_stats=rollout_policy.obs_normalization_stats,
        action_normalization_stats=rollout_policy.action_normalization_stats,
    )


class ObservationHistory:
    """Stacks the last ``num_frames`` observations, as robomimic's FrameStackWrapper.

    At the start of an episode the first observation is repeated. With
    ``num_frames == 1`` observations are passed through unchanged.
    """

    def __init__(self, num_frames):
        self.num_frames = int(num_frames)
        self._history = None

    def reset(self, obs):
        if self.num_frames > 1:
            self._history = {k: deque([v[None]] * self.num_frames, maxlen=self.num_frames) for k, v in obs.items()}
            return self.stacked()
        self._history = obs
        return obs

    def push(self, obs):
        if self.num_frames > 1:
            for k, v in obs.items():
                self._history[k].append(v[None])
            return self.stacked()
        self._history = obs
        return obs

    def stacked(self):
        return {k: np.concatenate(list(v), axis=0) for k, v in self._history.items()}
