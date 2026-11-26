"""Reverse diffusion with GRACE guidance (paper Algorithm 1)."""

import time

import torch

import robomimic.utils.tensor_utils as TensorUtils

from grace.guidance.ddpm import reverse_mean_std
from grace.guidance.posterior import grace_posterior_mean


def _cuda_sync(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class GraceSampler:
    """Samples action sequences U^(0) from a trained robomimic diffusion policy.

    The prior is a robomimic ``DiffusionPolicyUNet`` (UNet or residual-CNN
    noise predictor) with a DDPM scheduler. Its EMA network, noise scheduler
    and horizons are used as they are.

    With ``cfg`` and ``task`` given, the last ``cfg.N_g`` reverse steps are
    guided (Algorithm 1). Without them, plain DDPM sampling of the prior is
    performed (the unguided prior).

    Horizons (robomimic names in brackets): the network predicts H
    (prediction_horizon) actions, conditioned on T_o (observation_horizon)
    observations. The first T_o - 1 actions belong to past observation steps
    and are never executed; GRACE rolls out and guides the remaining
    H - T_o + 1 actions. H_exec (action_horizon) actions are executed per plan.
    """

    def __init__(self, algo, cfg=None, task=None, normalizer=None):
        if (cfg is None) != (task is None) or (task is not None and normalizer is None):
            raise ValueError("cfg, task and normalizer must be given together")
        algo_config = algo.algo_config
        if not algo_config.ddpm.enabled:
            raise NotImplementedError("GRACE is implemented for DDPM priors")
        self.algo = algo
        self.cfg = cfg
        self.task = task
        self.normalizer = normalizer
        self.T_o = int(algo_config.horizon.observation_horizon)
        self.H = int(algo_config.horizon.prediction_horizon)
        self.H_exec = int(algo_config.horizon.action_horizon)
        self.N = int(algo_config.ddpm.num_inference_timesteps)
        self.clip_sample = bool(algo_config.ddpm.clip_sample)
        if cfg is not None and not (1 <= int(cfg.N_g) <= self.N):
            raise ValueError("N_g must be in [1, N={}], got {}".format(self.N, cfg.N_g))
        # Wall-clock time of each guided step of the last call (Alg. 1, lines 6-12).
        self.last_guidance_times_ms = []

    @property
    def guided(self):
        return self.cfg is not None

    @torch.no_grad()
    def sample(self, obs):
        """Plans from a normalized observation dict {key: [1, T_o, D]}.

        Returns the H_exec actions to execute, [1, H_exec, m], normalized.
        """
        algo = self.algo
        assert not algo.nets.training
        nets = algo.ema.averaged_model if algo.ema is not None else algo.nets

        # Observation conditioning of the noise predictor.
        inputs = {"obs": obs, "goal": None}
        for k in algo.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(algo.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(algo.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3
        B = obs_features.shape[0]
        obs_cond = obs_features.flatten(start_dim=1)

        # Line 1: U^(N) ~ N(0, I).
        U = torch.randn((B, self.H, algo.ac_dim), device=algo.device)

        scheduler = algo.noise_scheduler
        scheduler.set_timesteps(self.N)
        rollout_cost = self.task.bind(inputs["obs"]) if self.guided else None
        s = self.T_o - 1  # first executed (and guided) index of the sequence
        self.last_guidance_times_ms = []

        # Line 2: i = N, ..., 1 (scheduler timestep t = i - 1).
        for step_idx, t in enumerate(scheduler.timesteps):
            eps = nets["policy"]["noise_pred_net"](sample=U, timestep=t, global_cond=obs_cond)

            if self.guided and step_idx >= self.N - int(self.cfg.N_g):  # line 5: i <= N_g
                # Line 3: reverse mean mu^(i); Sigma^(i) = std^2 I.
                mu, std = reverse_mean_std(scheduler, t, U, eps)
                # Lines 4, 6-12.
                _cuda_sync(U.device)
                tic = time.perf_counter()
                U_hat = mu.clone()
                U_hat[:, s:, :] = grace_posterior_mean(mu[:, s:, :], rollout_cost, self.normalizer, self.cfg)
                _cuda_sync(U.device)
                self.last_guidance_times_ms.append((time.perf_counter() - tic) * 1000.0)
                # Line 14: U^(i-1) ~ N(U_hat^(i-1), Sigma^(i)); no noise in the last step.
                if step_idx < self.N - 1:
                    U = U_hat + std * torch.randn_like(U_hat)
                else:
                    U = U_hat
                # Keep the iterate in the range of the training data, as the
                # scheduler does for the unguided steps (clip_sample).
                if self.clip_sample:
                    U = U.clamp(-1.0, 1.0)
            else:
                # Unguided reverse step (lines 3 and 14 with U_hat = mu).
                U = scheduler.step(model_output=eps, timestep=t, sample=U).prev_sample

        # Line 16: the first H_exec executable actions of U^(0).
        return U[:, s : s + self.H_exec]
