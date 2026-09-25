"""Parameter perturbation and batch scanning with full isolation.

Every perturbation builds a NEW immutable Network (the input is never
mutated), and every scan entry is simulated independently. Because
:mod:`app.simulation` keys its cache on a fingerprint that includes the
model version and all rate parameters, results from different versions
or parameter values can never be reused by mistake, while identical
requests remain bit-for-bit reproducible.
"""

from __future__ import annotations

import dataclasses

from .errors import ErrorCode, NetworkValidationError
from .models import Network, RateLaw
from .simulation import Trajectory, simulate


def perturb_parameter(
    network: Network, reaction: str, param: str, value: float
) -> Network:
    """Return a NEW network with one forward rate parameter replaced."""
    target = None
    for r in network.reactions:
        if r.name == reaction:
            target = r
            break
    if target is None:
        raise NetworkValidationError(
            ErrorCode.UNKNOWN_PARAMETER,
            f"no reaction named '{reaction}' in network",
        )
    if param not in target.rate.params:
        raise NetworkValidationError(
            ErrorCode.UNKNOWN_PARAMETER,
            f"reaction '{reaction}' has no forward parameter '{param}'",
        )
    new_rate = RateLaw(target.rate.kind, {**target.rate.params, param: value})
    new_reaction = dataclasses.replace(target, rate=new_rate)
    reactions = tuple(new_reaction if r is target else r for r in network.reactions)
    return dataclasses.replace(network, reactions=reactions)


def scan_parameter(
    network: Network,
    reaction: str,
    param: str,
    values: list[float],
    t_end: float,
    n_points: int = 101,
) -> list[Trajectory]:
    """Simulate the network once per parameter value, in isolation."""
    results = []
    for value in values:
        perturbed = perturb_parameter(network, reaction, param, value)
        results.append(simulate(perturbed, t_end=t_end, n_points=n_points))
    return results
