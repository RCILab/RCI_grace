"""Diffusion reverse kernel p_theta(U^(i-1) | U^(i)) = N(mu^(i), Sigma^(i)) (paper eqs. (3)-(4))."""

import torch


def reverse_mean_std(scheduler, t, U, eps):
    """Reverse mean mu^(i) and reverse standard deviation of a DDPM step.

    ``scheduler`` is the diffusers ``DDPMScheduler`` of the prior, ``t`` the
    scheduler timestep of reverse step i (t = i - 1 with N inference steps),
    ``U`` the current iterate U^(i) and ``eps`` = eps_theta(U^(i), i).

    mu^(i) is computed exactly as in ``DDPMScheduler.step``: through the
    predicted clean sample, which is clipped to [-1, 1] (``clip_sample``). This
    equals eq. (4) whenever the clip is inactive. The returned standard
    deviation is sqrt of the "fixed_small" variance (Sigma^(i) = std^2 I) and
    zero at the last step (t = 0).
    """
    config = scheduler.config
    if config.prediction_type != "epsilon" or scheduler.variance_type != "fixed_small":
        raise NotImplementedError("GRACE expects an epsilon-prediction DDPM with fixed_small variance")
    t = int(t)
    alpha_prod_t = scheduler.alphas_cumprod[t].to(device=U.device, dtype=U.dtype)
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[t - 1].to(device=U.device, dtype=U.dtype)
        if t > 0
        else scheduler.one.to(device=U.device, dtype=U.dtype)
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    U0_hat = (U - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()
    if config.clip_sample:
        U0_hat = torch.clamp(U0_hat, -1, 1)
    coeff_U0 = alpha_prod_t_prev.sqrt() * scheduler.betas[t].to(U.device, U.dtype) / beta_prod_t
    coeff_U = scheduler.alphas[t].to(U.device, U.dtype).sqrt() * beta_prod_t_prev / beta_prod_t
    mu = coeff_U0 * U0_hat + coeff_U * U

    if t == 0:
        std = torch.zeros((), device=U.device, dtype=U.dtype)
    else:
        variance = scheduler._get_variance(t)
        if not torch.is_tensor(variance):
            variance = torch.tensor(variance, device=U.device, dtype=U.dtype)
        std = variance.to(device=U.device, dtype=U.dtype).sqrt()
    return mu, std
