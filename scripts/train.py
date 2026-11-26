"""Train the diffusion prior (robomimic Diffusion Policy, UNet or residual-CNN noise predictor).

The prior is trained once, offline, on demonstrations without the deployment
obstacles; GRACE does not modify training.

    python scripts/train.py --config configs/train/pointmass2d_unet.json --dataset data/pointmass2d_10000.hdf5
    python scripts/train.py --config configs/train/panda_cnn.json --dataset data/panda_20000.hdf5 --output runs
"""

import argparse
import json
import os
import traceback

import _bootstrap  # noqa: F401

import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="robomimic config (configs/train/*.json)")
    parser.add_argument("--dataset", required=True, help="robomimic hdf5 dataset")
    parser.add_argument("--output", default=None, help="output directory (default: config train.output_dir)")
    parser.add_argument("--name", default=None, help="experiment name (default: config experiment.name)")
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint of the run")
    args = parser.parse_args()

    with open(args.config) as f:
        ext_cfg = json.load(f)
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
        config.train.data = [{"path": args.dataset}]
        # Relative output directories are taken from the current directory
        # (robomimic would resolve them against its own package directory).
        output_dir = args.output if args.output is not None else config.train.output_dir
        config.train.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        if args.name is not None:
            config.experiment.name = args.name
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    config.lock()
    try:
        train(config, device=device, resume=args.resume)
    except Exception as e:
        print("run failed with error:\n{}\n\n{}".format(e, traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
