"""Task models for GRACE guidance.

A task provides ``bind(obs) -> rollout_cost``: for the current (normalized)
observation it returns a function that maps candidate action sequences
V [K, H, m] (raw units) to the task part of the cost J [K] (eq. (31) without
the prior term), by rolling out the system model from the current state.
"""
