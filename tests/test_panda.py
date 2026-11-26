import numpy as np
import torch

from grace.envs.panda.capsules import PandaCapsuleFK, capsule_radii, self_collision_pairs
from grace.envs.panda.config import PandaConfig
from grace.eval.panda_scenario import ScenarioConfig, make_trial


def test_config_roundtrip():
    cfg = PandaConfig()
    assert cfg.substeps == 20 and cfg.episode_horizon == 1001
    assert PandaConfig.from_saved(str(cfg.to_dict())).to_dict().keys() == cfg.to_dict().keys()


def test_capsule_fk_shapes():
    fk = PandaCapsuleFK()
    start, end = fk.forward_collision_segments(torch.zeros(3, 7))
    n = len(capsule_radii())
    assert start.shape == (3, n, 3) and end.shape == (3, n, 3)
    assert len(self_collision_pairs()) > 0


def test_trial_construction():
    trial = make_trial(PandaConfig(), ScenarioConfig(), 0, 0)
    assert len(trial.obstacles) == 6
    assert trial.q_start.shape == (7,) and trial.q_goal.shape == (7,)
    assert np.linalg.norm(trial.goal_pos) > 0


def test_datagen_resume_matches_uninterrupted_run(tmp_path):
    from grace.datagen.panda.generate import generate_dataset, load_snapshot, save_snapshot

    config = PandaConfig(seed=0)
    full, resumed = tmp_path / "full.npz", tmp_path / "resumed.npz"
    generate_dataset(full, 4, config, num_workers=1)
    generate_dataset(resumed, 2, config, num_workers=1)  # stopped after two episodes, ...
    save_snapshot(resumed, load_snapshot(resumed, config)[1:], config, 4)  # ... the first one still running
    generate_dataset(resumed, 4, config, num_workers=2)
    a, b = load_snapshot(full, config), load_snapshot(resumed, config)
    assert [int(ep["attempt_index"]) for ep in a] == [int(ep["attempt_index"]) for ep in b]
    assert all(np.array_equal(x["observed_q"], y["observed_q"]) for x, y in zip(a, b))
