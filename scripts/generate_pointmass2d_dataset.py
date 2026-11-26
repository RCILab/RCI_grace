"""Generate the point-mass demonstration dataset (robomimic hdf5).

Paper setting (Table I):
    python scripts/generate_pointmass2d_dataset.py --output data/pointmass2d_10000.hdf5 --num-demos 10000 --num-workers 16 --seed 0
"""

import argparse
import json

import _bootstrap  # noqa: F401

from grace.datagen.pointmass2d.generate import generate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-demos", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    summary = generate_dataset(
        args.output, num_demos=args.num_demos, num_workers=args.num_workers, seed=args.seed, val_ratio=args.val_ratio
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
