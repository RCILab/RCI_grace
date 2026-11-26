# GRACE: Gradient-Free Robot Action Generation via Combined Diffusion-MPPI Posterior Mean Estimation

Inference-time guidance of a pretrained diffusion policy with an MPPI posterior mean: no retraining, no cost gradients.

## Overview
This repository is the simulation code of the paper *GRACE: Gradient-Free Robot Action Generation via Combined Diffusion-MPPI Posterior Mean Estimation* (IEEE Robotics and Automation Letters, 2026) by Leesai Park, Jiho Hong and Sanghyun Kim.  
Project page: https://rcilab.khu.ac.kr/grace/

GRACE steers a pretrained diffusion policy at inference time. At each of the last N_g reverse diffusion steps it:

1. samples K perturbations around the diffusion reverse mean;
2. rolls each perturbed action sequence through a system model and scores it with the deployment-time cost (binary collision and constraint terms included);
3. moves the reverse mean to the MPPI-weighted posterior mean, and draws the next iterate with the diffusion-schedule covariance.

The release covers the two simulated tasks of the paper, each with demonstration generation, training of the diffusion prior and GRACE evaluation:

| | Point mass (Sec. IV, Table II) | FR3 goal reaching (Sec. IV, Table IV) |
|---|---|---|
| Action | 2D velocity | joint increment Δq (torque-controlled arm in MuJoCo) |
| Demonstrations | 10,000 OMPL RRTConnect paths, base obstacles only | 20,000 obstacle-free feedback-tracking reaches |
| Evaluation | 30 random start/goal pairs, base + unseen obstacles | 30 random start/goal pairs, 6 random unseen obstacles |

The real-robot experiment, the baselines, the datasets and the trained checkpoints are not part of this release.

## Dependencies
Tested with Python 3.10 on Ubuntu 22.04 with an NVIDIA GPU (CUDA 12.8). ROS is not required.

- [PyTorch](https://pytorch.org) 2.8.0
- [diffusers](https://github.com/huggingface/diffusers) 0.11.1 (later versions change the DDPM scheduler and EMA API)
- [robomimic](https://github.com/ARISE-Initiative/robomimic) v0.5.0, bundled in `third_party/robomimic` (see [NOTICE](NOTICE))
- [MuJoCo](https://github.com/google-deepmind/mujoco) 3.3.4 and [Pinocchio](https://github.com/stack-of-tasks/pinocchio) 2.7.0 (`pin`): FR3 task only
- [OMPL](https://ompl.kavrakilab.org) 2.0.1: point-mass demonstration generation only

All versions are pinned in [requirements.txt](requirements.txt).

## Installation
```bash
git clone https://github.com/RCILab/RCI_grace.git
cd RCI_grace
conda create -n grace -c conda-forge python=3.10 -y
conda activate grace
pip install -r requirements.txt
```
All dependencies are installed from PyPI with pip inside the conda environment. If ROS is sourced in your shell, run `unset PYTHONPATH` first so that its Python packages do not shadow the pinned ones.

The scripts add the repository root and `third_party/` to the Python path themselves; no package installation step is needed.

To check the installation:
```bash
pip install pytest
python -m pytest tests
```

## Usage
Each task follows the same three steps. Commands are run from the repository root.

```bash
# 1. demonstrations
python scripts/generate_pointmass2d_dataset.py --output <dataset>.hdf5 ...
python scripts/generate_panda_dataset.py --output <dataset>.npz ...      # then --convert to hdf5

# 2. diffusion prior (robomimic Diffusion Policy, UNet or residual 1D CNN)
python scripts/train.py --config configs/train/<task>_<unet|cnn>.json --dataset <dataset>.hdf5

# 3. evaluation with GRACE (add --no-guidance for the unguided prior)
python scripts/eval_<task>.py configs/eval/<task>_grace_<unet|cnn>.yaml --checkpoint <ckpt>.pth
```

- **Resuming.** Both demonstration generators can be stopped (Ctrl+C) and resumed by running the same command again. The result is identical to an uninterrupted run. Training resumes the latest run with `--resume`.
- **Overrides.** The evaluation scripts accept `--output-dir`, `--seed`, `--num-episodes` / `--num-trials` and, for the FR3, `--torch-seed`. Run a script with `--help` for all options.
- **Outputs.** Each evaluation writes the following files to `output_dir`:
  - `summary.json`: the success count, failures by type, the success-only path length (mean and population std, as in Table II), and planning and guidance times;
  - `trajectories.npz`: the executed trajectories.

## Examples
1. **Point mass**: 2D navigation among unseen obstacles
   ```bash
   # 10,000 demonstrations on the base map (needs OMPL)
   python scripts/generate_pointmass2d_dataset.py --output data/pointmass2d_10000.hdf5 --num-demos 10000 --num-workers 16

   python scripts/train.py --config configs/train/pointmass2d_unet.json --dataset data/pointmass2d_10000.hdf5

   # 30 episodes with the unseen obstacles
   python scripts/eval_pointmass2d.py configs/eval/pointmass2d_grace_unet.yaml --checkpoint runs/pointmass2d_unet/<run>/models/<ckpt>.pth
   python scripts/eval_pointmass2d.py configs/eval/pointmass2d_grace_unet.yaml --checkpoint <ckpt>.pth --no-guidance
   ```
2. **FR3 goal reaching**: torque-controlled 7-DoF arm among 6 random unseen obstacles
   ```bash
   # 20,000 obstacle-free demonstrations, then conversion to robomimic hdf5
   python scripts/generate_panda_dataset.py --output data/panda_20000.npz --episodes 20000 --workers 16
   python scripts/generate_panda_dataset.py --convert data/panda_20000.npz --hdf5 data/panda_20000.hdf5

   python scripts/train.py --config configs/train/panda_unet.json --dataset data/panda_20000.hdf5

   # 30 trials
   python scripts/eval_panda.py configs/eval/panda_grace_unet.yaml --checkpoint runs/panda_unet/<run>/models/<ckpt>.pth
   ```
Replace `unet` with `cnn` for the residual 1D-CNN prior.

## Repository layout
```
grace/
  guidance/        GRACE (Algorithm 1), independent of the task
    config.py        GraceConfig: N_g, K, lam, sigma_g, w_prior
    ddpm.py          reverse mean and std of the DDPM reverse kernel (eqs. (3)-(4))
    posterior.py     MPPI estimate of the guidance posterior mean (eq. (24), Alg. 1 lines 6-12)
    sampler.py       guided reverse diffusion (Alg. 1)
    normalization.py normalized <-> raw actions and observations
  tasks/           rollout models and costs used by the guidance
    pointmass2d.py   x_{t+1} = x_t + clip(v_t) dt, J_obs (eq. (30))
    panda.py         controller-aware joint rollout, J_safe (obstacle / self / floor / joint limit)
  envs/            simulation (point-mass map and environment; FR3 MuJoCo model, kinematics, capsules, obstacles)
  datagen/         demonstrations (OMPL expert; FR3 feedback joint-goal expert and conversion to hdf5)
  eval/            synchronous evaluation of both tasks
  policy.py        checkpoint loading, closed-loop execution of H_exec actions per plan
configs/           training (robomimic json) and evaluation (yaml) configs with the paper settings
scripts/           command-line entry points
third_party/       robomimic v0.5.0 (Diffusion Policy)
tests/
```
A task plugs into the guidance by providing `bind(obs) -> rollout_cost`, where `rollout_cost` maps K raw action sequences to K costs.

## GRACE in the code
| Algorithm 1 | Code |
|---|---|
| 1: U^(N) ~ N(0, I) | `GraceSampler.sample` |
| 3: μ^(i) = μ_θ(U^(i), i) | `ddpm.reverse_mean_std` |
| 5: guided if i ≤ N_g | `GraceSampler.sample` (`step_idx >= N - N_g`) |
| 7: δU_k ~ N(0, Σ_g) | `posterior.grace_posterior_mean` (`delta_U`) |
| 8: Ū_k = μ^(i) + δU_k | `U_bar` |
| 9: roll out, evaluate J | `task.bind(obs)` → `rollout_cost(V)`, plus `w_prior * J_prior` |
| 11: w_k ∝ exp(−J_k / λ) | `posterior.mppi_weights` |
| 12: Û = μ^(i) + Σ_k w_k δU_k | `U_hat` |
| 14: U^(i−1) ~ N(Û, Σ^(i)) | `GraceSampler.sample` (no noise at the last step; clipped to [−1, 1]) |
| 16: execute H_exec actions, replan | `policy.RecedingHorizonPlanner` (point mass), `eval/panda.run_trial` (FR3) |

## Paper settings
| | Point mass | FR3 |
|---|---|---|
| Prior (robomimic Diffusion Policy) | H = 16, T_o = 2, H_exec = 8 | H = 32, T_o = 1, H_exec = 16 |
| DDPM | N = 100 (squaredcos_cap_v2, ε-prediction) | same |
| N_g, K | 10, 1000 | 5, 1000 |
| λ | 0.5 | 0.01 |
| sigma_g (normalized) | 0.7 | 0.5 |
| w_prior | 1000 | 100 |
| Cost weights | w_obs = 1000 | w_obs = w_self = w_jl = 2000, w_floor = 1600 |
| Step budget / goal tolerance | 1000 / 0.05 m | 5000 / 0.03 rad |

All values are in `configs/eval/*.yaml` and `configs/train/*.json`.

## Citation
If you use this code in your research, please cite:

```bibtex
@article{park2026grace,
  title   = {{GRACE}: Gradient-Free Robot Action Generation via Combined Diffusion-{MPPI} Posterior Mean Estimation},
  author  = {Park, Leesai and Hong, Jiho and Kim, Sanghyun},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Third-party components (robomimic, the MuJoCo Menagerie Panda model, the franka_description URDF) keep their own licenses; see [NOTICE](NOTICE).

## Contact
Maintainer: Leesai Park (leesai2000@khu.ac.kr)  
Lab: [RCI Lab @ Kyung Hee University](https://rcilab.khu.ac.kr)
