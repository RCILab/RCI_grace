"""Conversion between the prior's normalized space and raw units.

robomimic stores the min-max normalization of actions and the observation
normalization as affine maps, x_raw = x_norm * scale + offset. GRACE samples
candidates in the normalized action space (where the prior lives) and rolls
them out in raw units.
"""

import torch


def _f32(value, device):
    return torch.tensor(value, dtype=torch.float32, device=device).view(-1)


class AffineNormalizer:
    def __init__(self, action_offset, action_scale, obs_offset, obs_scale):
        self.action_offset = action_offset  # [m]
        self.action_scale = action_scale  # [m]
        self.obs_offset = obs_offset  # {key: [D]}
        self.obs_scale = obs_scale  # {key: [D]}

    @classmethod
    def from_rollout_policy(cls, rollout_policy, device):
        """Uses the statistics stored in the checkpoint (``policy_from_checkpoint``)."""
        action_stats = rollout_policy.action_normalization_stats
        if action_stats is None or len(action_stats) != 1:
            raise ValueError("Expected exactly one normalized action key in the checkpoint")
        entry = next(iter(action_stats.values()))
        if not {"offset", "scale"} <= set(entry):
            raise ValueError("Expected affine (offset/scale) action normalization statistics")
        obs_offset, obs_scale = {}, {}
        for key, obs_entry in (rollout_policy.obs_normalization_stats or {}).items():
            obs_offset[key] = _f32(obs_entry["offset"], device)
            obs_scale[key] = _f32(obs_entry["scale"], device)
        return cls(_f32(entry["offset"], device), _f32(entry["scale"], device), obs_offset, obs_scale)

    def unnormalize_actions(self, U_T):
        """U_T: [K, m, H] normalized action sequences (action dimension in the middle)."""
        return U_T * self.action_scale.view(1, -1, 1) + self.action_offset.view(1, -1, 1)

    def unnormalize_obs(self, key, x):
        """x: [1, D] normalized observation of ``key``."""
        return x * self.obs_scale[key].view(1, -1) + self.obs_offset[key].view(1, -1)
