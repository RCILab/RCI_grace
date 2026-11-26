import h5py
import numpy as np
import pytest

from grace.envs.pointmass2d import ENV_ID_EVAL, PointMassConfig, make_dense2d_map, make_env


def test_map_sizes():
    m = make_dense2d_map()
    assert m.base_sphere_centers.shape == (14, 2) and m.base_box_centers.shape == (15, 2)
    assert m.extra_sphere_centers.shape == (5, 2) and m.extra_box_centers.shape == (3, 2)


def test_reset_is_deterministic():
    starts = []
    for _ in range(2):
        env = make_env(ENV_ID_EVAL, PointMassConfig().env_kwargs(), seed=0)
        starts.append([env.reset()["position"].copy() for _ in range(3)])
    assert np.array_equal(np.asarray(starts[0]), np.asarray(starts[1]))


def test_step_moves_by_clipped_velocity():
    env = make_env(ENV_ID_EVAL, PointMassConfig().env_kwargs(), seed=0)
    env.reset()
    env.set_state({"position": np.zeros(2, np.float32), "velocity": np.zeros(2, np.float32), "goal": np.ones(2, np.float32), "step": 0})
    obs, _, _, info = env.step(np.array([10.0, 0.0]))
    if not info["collision"]:
        assert np.allclose(obs["position"], [0.5 * 0.02, 0.0])


def test_datagen_resume_matches_uninterrupted_run(tmp_path):
    pytest.importorskip("ompl")
    from grace.datagen.pointmass2d.generate import generate_worker

    kwargs = PointMassConfig().env_kwargs()

    def start_goal_pairs(path):
        with h5py.File(path, "r") as f:
            demos = [f["data"]["demo_{}".format(i)] for i in range(len(f["data"]))]
            return [np.r_[d.attrs["start"], d.attrs["goal"]].tolist() for d in demos]

    full = str(tmp_path / "full.hdf5")
    generate_worker(0, 4, kwargs, 0, 100, full)

    resumed = str(tmp_path / "resumed.hdf5")
    generate_worker(0, 2, kwargs, 0, 100, resumed)  # stopped after two demonstrations ...
    with h5py.File(resumed, "a") as f:  # ... while a third one was being written
        f["data"].create_group("demo_2").create_dataset("actions", data=np.zeros((1, 2), np.float32))
    generate_worker(0, 4, kwargs, 0, 100, resumed)
    assert start_goal_pairs(resumed) == start_goal_pairs(full)
