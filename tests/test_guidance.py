import torch

from grace.guidance import GraceConfig, grace_posterior_mean, mppi_weights


class _IdentityNormalizer:
    def unnormalize_actions(self, U_T):
        return U_T


def test_mppi_weights_normalized_and_ordered():
    J = torch.tensor([3.0, 1.0, 2.0])
    w = mppi_weights(J, lam=0.5)
    assert torch.isclose(w.sum(), torch.tensor(1.0))
    assert int(torch.argmax(w)) == 1


def test_posterior_mean_shapes_and_time_tied_samples():
    torch.manual_seed(0)
    H, m, K = 15, 2, 64
    mu = torch.rand(1, H, m) * 2 - 1
    seen = {}

    def cost(V):  # V: [K, H, m]
        seen["V"] = V.clone()
        return torch.zeros(V.shape[0])

    cfg = GraceConfig(N_g=10, K=K, lam=0.5, sigma_g=0.7, w_prior=1000.0)
    U_hat = grace_posterior_mean(mu, cost, _IdentityNormalizer(), cfg)
    assert U_hat.shape == mu.shape
    V = seen["V"]
    assert V.shape == (K, H, m)
    # sample 0 is the reverse mean itself
    assert torch.allclose(V[0], mu[0])
    # each perturbation is constant over the horizon (time-tied)
    delta = V - mu
    assert torch.allclose(delta, delta[:, :1, :].expand_as(delta), atol=1e-6)


def test_low_temperature_selects_cheapest_sample():
    torch.manual_seed(1)
    H, m, K = 8, 2, 32
    mu = torch.zeros(1, H, m)
    target = torch.tensor([0.3, -0.2])

    def cost(V):
        return ((V[:, 0, :] - target) ** 2).sum(dim=-1) * 1e4

    cfg = GraceConfig(N_g=1, K=K, lam=1e-6, sigma_g=0.5, w_prior=0.0)
    U_hat = grace_posterior_mean(mu, cost, _IdentityNormalizer(), cfg)
    assert torch.allclose(U_hat[0, 0], U_hat[0, -1])  # still time-tied


def test_rng_compat_draw():
    mu = torch.zeros(1, 4, 2)
    cfg = dict(N_g=1, K=16, lam=0.5, sigma_g=0.5, w_prior=1.0)
    torch.manual_seed(0)
    grace_posterior_mean(mu, lambda V: torch.zeros(V.shape[0]), _IdentityNormalizer(), GraceConfig(**cfg, paper_rng_compat=True))
    a = torch.rand(1)
    torch.manual_seed(0)
    grace_posterior_mean(mu, lambda V: torch.zeros(V.shape[0]), _IdentityNormalizer(), GraceConfig(**cfg, paper_rng_compat=False))
    b = torch.rand(1)
    assert not torch.equal(a, b)
