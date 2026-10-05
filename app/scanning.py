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
from dataclasses import dataclass

from .errors import ErrorCode, NetworkError, NetworkValidationError
from .models import Network, RateLaw
from .simulation import Trajectory, network_fingerprint, simulate


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


@dataclass(frozen=True)
class ScanOutcome:
    """One scan point: either a trajectory or the locatable error that
    prevented it. A failed point never aborts or contaminates the rest."""

    value: float
    fingerprint: str
    trajectory: Trajectory | None
    error: NetworkError | None


def scan_parameter_isolated(
    network: Network,
    reaction: str,
    param: str,
    values: list[float],
    t_end: float,
    n_points: int = 101,
) -> list[ScanOutcome]:
    """Scan one parameter, isolating per-value failures.

    Unknown reaction/parameter names still raise immediately (they are
    independent of the scanned value); a network that only fails to build
    or simulate at specific values yields an error outcome for exactly
    those points while the remaining points return normal trajectories.
    """
    outcomes = []
    for value in values:
        perturbed = perturb_parameter(network, reaction, param, value)
        fingerprint = network_fingerprint(perturbed)
        try:
            trajectory = simulate(perturbed, t_end=t_end, n_points=n_points)
        except NetworkError as exc:
            outcomes.append(
                ScanOutcome(value, fingerprint, trajectory=None, error=exc)
            )
        else:
            outcomes.append(
                ScanOutcome(value, fingerprint, trajectory, error=None)
            )
    return outcomes
