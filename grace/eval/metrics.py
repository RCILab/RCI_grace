"""Summary statistics as reported in the paper."""

import numpy as np


def summarize_episodes(episodes):
    """Success / failure counts, success-only path length (Table II) and planning times (Table IV).

    ``episodes``: list of dicts with ``success`` (bool), ``failure`` (str or
    None) and optionally ``path_length``. Path length statistics use the
    successful episodes only; the standard deviation is the population std.
    """
    n = len(episodes)
    successes = [ep for ep in episodes if ep["success"]]
    failures = {}
    for ep in episodes:
        if not ep["success"]:
            failures[ep["failure"]] = failures.get(ep["failure"], 0) + 1
    summary = {
        "num_episodes": n,
        "num_success": len(successes),
        "success_rate": (len(successes) / n) if n else 0.0,
        "failures": failures,
    }
    lengths = [ep["path_length"] for ep in successes if ep.get("path_length") is not None]
    if lengths:
        summary["path_length_success_mean"] = float(np.mean(lengths))
        summary["path_length_success_std"] = float(np.std(lengths))
    plan_ms = [t for ep in episodes for t in ep.get("planning_times_ms", [])]
    if plan_ms:
        summary["planning_time_mean_ms"] = float(np.mean(plan_ms))
        summary["planning_time_std_ms"] = float(np.std(plan_ms))
    guide_ms = [t for ep in episodes for t in ep.get("guidance_plan_times_ms", [])]
    if guide_ms:
        summary["guidance_time_per_plan_mean_ms"] = float(np.mean(guide_ms))
        summary["guidance_time_per_plan_std_ms"] = float(np.std(guide_ms))
    return summary
