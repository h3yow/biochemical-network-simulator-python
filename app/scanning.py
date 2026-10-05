"""Parameter perturbation and batch scanning with full isolation.

Every perturbation builds a NEW immutable Network (the input is never
mutated), and every scan entry is simulated independently. Because
:mod:`app.simulation` keys its cache on a fingerprint that includes the
model version and all rate parameters (and composition, when present),
results from different versions or parameter values can never be reused
by mistake, while identical requests remain bit-for-bit reproducible.

:func:`safe_scan_parameter` additionally isolates *failures*: one
parameter value that cannot build or simulate a network is reported at
that point with its machine-readable code and detail, while every other
point still returns its trajectory.
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
class ScanPoint:
    """One scan entry: either a trajectory or a locatable failure."""

    value: float
    ok: bool
    fingerprint: str | None = None
    trajectory: Trajectory | None = None
    error_code: str | None = None
    error_detail: str | None = None


def safe_scan_parameter(
    network: Network,
    reaction: str,
    param: str,
    values: list[float],
    t_end: float,
    n_points: int = 101,
    conservation_tol: float = 1e-6,
) -> list[ScanPoint]:
    """Scan with per-point failure isolation.

    Each value gets its own freshly built immutable network and its own
    simulation. A value that fails to build or to simulate (including a
    conservation violation at that value) yields a failed
    :class:`ScanPoint` carrying the original ``ErrorCode`` and detail --
    which already names the reaction, species, component or conservation
    law responsible -- and never prevents the remaining points from
    running. Concurrent scans cannot share state: networks are
    immutable and the solution cache is keyed per fingerprint.
    """
    points: list[ScanPoint] = []
    for value in values:
        try:
            perturbed = perturb_parameter(network, reaction, param, value)
            trajectory = simulate(
                perturbed,
                t_end=t_end,
                n_points=n_points,
                conservation_tol=conservation_tol,
            )
            points.append(
                ScanPoint(
                    value=value,
                    ok=True,
                    fingerprint=network_fingerprint(perturbed),
                    trajectory=trajectory,
                )
            )
        except NetworkError as exc:
            points.append(
                ScanPoint(
                    value=value,
                    ok=False,
                    error_code=exc.code.value,
                    error_detail=f"value {value}: {exc.detail}",
                )
            )
    return points
