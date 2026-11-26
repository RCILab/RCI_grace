"""Generate the FR3 demonstration dataset (npz) and convert it to robomimic hdf5.

Paper setting (Table I):
    python scripts/generate_panda_dataset.py --output data/panda_20000.npz --episodes 20000 --workers 16
    python scripts/generate_panda_dataset.py --convert data/panda_20000.npz --hdf5 data/panda_20000.hdf5
"""

import argparse
import json

import _bootstrap  # noqa: F401

from grace.datagen.panda.convert import convert
from grace.datagen.panda.generate import generate_dataset
from grace.envs.panda.config import PandaConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", help="npz file to collect into (resumes if it exists)")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--convert", metavar="NPZ", help="convert this npz to robomimic hdf5 (--hdf5)")
    parser.add_argument("--hdf5", help="output hdf5 of --convert")
    parser.add_argument("--split-seed", type=int, default=20, help="seed of the train/valid split (paper: 20)")
    args = parser.parse_args()

    if args.convert:
        if not args.hdf5:
            parser.error("--convert needs --hdf5")
        print(json.dumps(convert(args.convert, args.hdf5, seed=args.split_seed), indent=2))
        return
    if not args.output:
        parser.error("--output is required")
    generate_dataset(args.output, args.episodes, PandaConfig(seed=args.seed), num_workers=args.workers, save_every=args.save_every)


if __name__ == "__main__":
    main()
