"""Obstacle map of the planar point-mass task (paper Fig. 3a).

The map follows the dense 2D environment of MPD [18]. ``base`` obstacles are
present when the demonstrations are generated (training); ``extra`` obstacles
(red in Fig. 3a) are only added at inference time.

All values are float32 and identical to the arrays used in the paper.
"""

from dataclasses import dataclass

import numpy as np


WORLD_LIMITS = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)


def _f32(values):
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class PointMassMap:
    """Discs (centers, radii) and axis-aligned boxes (centers, full sizes)."""

    limits: np.ndarray
    base_sphere_centers: np.ndarray
    base_sphere_radii: np.ndarray
    base_box_centers: np.ndarray
    base_box_sizes: np.ndarray
    extra_sphere_centers: np.ndarray
    extra_sphere_radii: np.ndarray
    extra_box_centers: np.ndarray
    extra_box_sizes: np.ndarray

    def all_sphere_centers(self, include_extra=False):
        if include_extra and self.extra_sphere_centers.size > 0:
            return np.concatenate([self.base_sphere_centers, self.extra_sphere_centers], axis=0)
        return self.base_sphere_centers

    def all_sphere_radii(self, include_extra=False):
        if include_extra and self.extra_sphere_radii.size > 0:
            return np.concatenate([self.base_sphere_radii, self.extra_sphere_radii], axis=0)
        return self.base_sphere_radii

    def all_box_centers(self, include_extra=False):
        if include_extra and self.extra_box_centers.size > 0:
            return np.concatenate([self.base_box_centers, self.extra_box_centers], axis=0)
        return self.base_box_centers

    def all_box_sizes(self, include_extra=False):
        if include_extra and self.extra_box_sizes.size > 0:
            return np.concatenate([self.base_box_sizes, self.extra_box_sizes], axis=0)
        return self.base_box_sizes


_DENSE2D = dict(
    base_sphere_centers=[
        [-0.43378472, 0.33346438],
        [0.33134747, 0.6288051],
        [-0.5656965, -0.4849945],
        [0.42124248, -0.6656166],
        [0.05636655, -0.5149664],
        [-0.38461784, -0.12315541],
        [0.80878216, 0.528787],
        [-0.013786113, 0.47400692],
        [0.13044947, -0.13676023],
        [0.62841374, -0.43461448],
        [0.17965621, -0.89262766],
        [0.67759687, 0.8817358],
        [-0.36087668, 0.83134586],
        [-0.115, -0.11],
    ],
    base_sphere_radii=[0.125] * 13 + [0.075],
    base_box_centers=[
        [0.6077817, 0.19512387],
        [0.5575313, 0.5508843],
        [-0.33522955, -0.68875194],
        [-0.6572632, 0.31827882],
        [-0.6645943, -0.016457155],
        [0.8165989, -0.19856024],
        [-0.8222246, -0.644858],
        [-0.30059898, -0.38341486],
        [-0.8946458, 0.89624476],
        [-0.23994406, 0.6021061],
        [-0.0061935885, 0.8456172],
        [0.3201036, -0.38119903],
        [-0.12204008, 0.14689502],
        [0.71563786, -0.6923346],
        [-0.6, -0.85],
    ],
    base_box_sizes=[[0.2, 0.2]] * 14 + [[0.1, 0.25]],
    extra_sphere_centers=[
        [-0.41, 0.1],
        [-0.075, -0.85],
        [-0.8240217, -0.33349368],
        [-0.6359214, 0.6683125],
        [0.1605742, 0.17920497],
    ],
    extra_sphere_radii=[0.075, 0.1, 0.1, 0.125, 0.125],
    extra_box_centers=[
        [0.46499997, -0.1],
        [0.36499998, 0.36499998],
        [-0.63, -0.23],
    ],
    extra_box_sizes=[
        [0.2, 0.2],
        [0.1, 0.15],
        [0.15, 0.1],
    ],
)


def make_dense2d_map():
    return PointMassMap(limits=WORLD_LIMITS.copy(), **{k: _f32(v) for k, v in _DENSE2D.items()})
