"""Hyperparameters of GRACE (Algorithm 1)."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GraceConfig:
    """Hyperparameters of Algorithm 1, in the notation of the paper.

    All action quantities are expressed in the normalized action space of the
    diffusion prior (robomimic min-max normalization to [-1, 1]).

    N_g      Number of guided reverse steps. Guidance is applied for i <= N_g,
             i.e. on the last N_g of the N reverse steps (Alg. 1, line 5).
    K        Number of MPPI samples per guided step (Alg. 1, line 6). Sample
             k = 1 is always the unperturbed reverse mean (delta = 0).
    lam      MPPI temperature lambda (Alg. 1, line 11).
    sigma_g  Exploration standard deviation: Sigma_g^(i) = sigma_g^2 * I for
             every guided step i (Alg. 1, line 7). It is independent of the
             diffusion-schedule covariance Sigma^(i). One perturbation per
             action dimension is shared by all time steps of the sequence
             ("same control perturbation across the time horizon", Sec. IV-A).
    w_prior  Weight of J_prior in eq. (31).

    paper_rng_compat
             The paper runs recorded debug traces, which drew one extra
             ``torch.randperm(K)`` from the global torch generator at every
             guided step (after line 12, before line 14). Keeping this draw
             reproduces the random stream of the paper experiments exactly; it
             has no other effect. Set to False for new experiments.
    """

    N_g: int
    K: int
    lam: float
    sigma_g: float
    w_prior: float
    paper_rng_compat: bool = True

    def to_dict(self):
        return asdict(self)


# Paper settings (Table I and Sec. IV).
POINTMASS2D = GraceConfig(N_g=10, K=1000, lam=0.5, sigma_g=0.7, w_prior=1000.0)
PANDA = GraceConfig(N_g=5, K=1000, lam=0.01, sigma_g=0.5, w_prior=100.0)
