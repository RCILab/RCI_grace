"""MPPI estimate of the guidance posterior mean (paper eq. (24); Algorithm 1, lines 6-12)."""

import torch


def mppi_weights(J, lam):
    """Alg. 1, line 11: w_k = exp(-J_k / lam) / sum_l exp(-J_l / lam).

    The minimum cost is subtracted before exponentiation (it cancels in the
    ratio) and small constants guard against division by zero.
    """
    w = torch.exp(-(J - torch.min(J)) / (float(lam) + 1e-8))
    return w / (w.sum() + 1e-10)


def grace_posterior_mean(mu, rollout_cost, normalizer, cfg):
    """Guided reverse mean U_hat^(i-1) for one guided reverse step.

    Args:
        mu: [1, H, m] reverse mean mu^(i) of the guided part of the action
            sequence, in the normalized action space of the prior.
        rollout_cost: callable V -> J_task, where V [K, H, m] are applied
            (raw) action sequences and J_task [K] is the task part of the cost
            (w_obs * J_obs for the point mass, J_safe for the FR3), i.e. eq.
            (31) without the prior term.
        normalizer: ``AffineNormalizer`` of the prior.
        cfg: ``GraceConfig``.

    Returns:
        U_hat: [1, H, m] = mu^(i) + sum_k w_k * delta_U_k   (Alg. 1, line 12).
    """
    device = mu.device
    B, H, m = mu.shape
    if B != 1:
        raise ValueError("GRACE guidance expects a batch size of 1, got {}".format(B))
    K = int(cfg.K)
    # Work with [K, m, H] tensors (action dimension in the middle).
    mu_T = mu[0:1, :, :m].transpose(1, 2)

    # Line 7: delta_U_k ~ N(0, Sigma_g), Sigma_g = sigma_g^2 I. One offset per
    # action dimension, shared by all H steps (time-tied perturbation).
    sigma_g = torch.as_tensor(float(cfg.sigma_g), device=device, dtype=torch.float32)
    delta_U = torch.randn((K, m, 1), device=device, dtype=torch.float32) * sigma_g
    # The first sample keeps the unperturbed reverse mean as a candidate.
    delta_U[0].zero_()
    delta_U = delta_U.expand(-1, -1, H).clone()

    # Line 8: U_bar_k = mu^(i) + delta_U_k.
    U_bar = mu_T.repeat(K, 1, 1) + delta_U

    # Line 9: roll out and evaluate J (eq. (31)). The task cost is evaluated on
    # the applied actions V (raw units) ...
    V = normalizer.unnormalize_actions(U_bar).transpose(1, 2)
    J_task = rollout_cost(V)
    # ... and J_prior (eq. (32)) on the normalized sequence. J_prior is the
    # mean of the squared deviation over the H*m entries, i.e. eq. (32) divided
    # by H*m; the value of w_prior refers to this normalized form.
    deviation = U_bar - mu_T
    J_prior = torch.mean(deviation * deviation, dim=(1, 2))
    J = J_task + float(cfg.w_prior) * J_prior

    # Lines 11-12.
    w = mppi_weights(J, cfg.lam)
    U_hat_T = mu_T + torch.sum(w.view(K, 1, 1) * delta_U, dim=0, keepdim=True)

    if cfg.paper_rng_compat:
        # See GraceConfig.paper_rng_compat: the paper runs drew this
        # permutation for their debug traces. The result is not used.
        torch.randperm(K, device=device)

    U_hat = mu.clone()
    U_hat[0:1, :, :m] = U_hat_T.transpose(1, 2)
    return U_hat
