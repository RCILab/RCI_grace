"""Evaluate GRACE (or the unguided prior) in the planar point-mass task.

    python scripts/eval_pointmass2d.py configs/eval/pointmass2d_grace_unet.yaml --checkpoint <prior.pth>
    python scripts/eval_pointmass2d.py configs/eval/pointmass2d_grace_unet.yaml --checkpoint <prior.pth> --no-guidance
"""

import argparse

import yaml

import _bootstrap  # noqa: F401

from grace.eval.pointmass2d import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="evaluation config (yaml)")
    parser.add_argument("--checkpoint", help="trained prior (.pth); overrides the config")
    parser.add_argument("--output-dir", help="overrides the config")
    parser.add_argument("--num-episodes", type=int, help="overrides the config")
    parser.add_argument("--seed", type=int, help="overrides the config")
    parser.add_argument("--no-guidance", action="store_true", help="sample the unguided prior")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.num_episodes is not None:
        config["num_episodes"] = args.num_episodes
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_guidance:
        config["guidance"] = None
    if not config.get("checkpoint"):
        parser.error("no checkpoint given (config 'checkpoint' or --checkpoint)")
    evaluate(config)


if __name__ == "__main__":
    main()
