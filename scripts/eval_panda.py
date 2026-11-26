"""Evaluate GRACE (or the unguided prior) in the FR3 goal-reaching task.

    python scripts/eval_panda.py configs/eval/panda_grace_unet.yaml --checkpoint <prior.pth>
    python scripts/eval_panda.py configs/eval/panda_grace_unet.yaml --checkpoint <prior.pth> --no-guidance
"""

import argparse

import yaml

import _bootstrap  # noqa: F401

from grace.eval.panda import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="evaluation config (yaml)")
    parser.add_argument("--checkpoint", help="trained prior (.pth); overrides the config")
    parser.add_argument("--output-dir", help="overrides the config")
    parser.add_argument("--num-trials", type=int, help="overrides the config")
    parser.add_argument("--seed", type=int, help="scenario seed; overrides the config")
    parser.add_argument("--torch-seed", type=int, help="seed of the sampling noise; overrides the config")
    parser.add_argument("--no-guidance", action="store_true", help="sample the unguided prior")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in ("checkpoint", "output_dir", "num_trials", "seed", "torch_seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.no_guidance:
        config["guidance"] = None
    if not config.get("checkpoint"):
        parser.error("no checkpoint given (config 'checkpoint' or --checkpoint)")
    evaluate(config)


if __name__ == "__main__":
    main()
