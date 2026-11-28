"""Solve an inertial interval conditioned on both endpoint states."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X


@dataclass(frozen=True)
class InertialState:
    pose: gtsam.Pose3
    velocity: np.ndarray
    bias: gtsam.imuBias.ConstantBias

    @classmethod
    def from_values(cls, values: gtsam.Values, node: int) -> InertialState:
        return cls(values.atPose3(X(node)), values.atVector(V(node)), values.atConstantBias(B(node)))


class ConditionalInertialError(ValueError):
    """The final interval graph error failed its numerical acceptance check."""


def solve_fixed_endpoints(graph: gtsam.NonlinearFactorGraph, initial_states: Sequence[InertialState],
                          settings: gtsam.LevenbergMarquardtParams, *,
                          error_increase_tolerance: float = 0.) -> tuple[list[InertialState], float]:
    """Add fixed endpoints to an interval graph with local X/V/B keys starting at zero."""
    values = gtsam.Values()
    for node, state in enumerate(initial_states):
        values.insert(X(node), state.pose)
        values.insert(V(node), state.velocity)
        values.insert(B(node), state.bias)
    last = len(initial_states) - 1
    for node in (0, last):
        state = initial_states[node]
        graph.add(gtsam.NonlinearEqualityPose3(X(node), state.pose))
        graph.add(gtsam.NonlinearEqualityPoint3(V(node), state.velocity))
        graph.add(gtsam.NonlinearEqualityConstantBias(B(node), state.bias))
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, settings)
    initial_error = optimizer.error()
    result = optimizer.optimize()
    final_error = optimizer.error()
    if not np.isfinite(final_error) or final_error > initial_error + error_increase_tolerance:
        raise ConditionalInertialError(f'区间目标函数从 {initial_error:.9g} 变为 {final_error:.9g}')
    return [InertialState.from_values(result, node) for node in range(1, last)], final_error
