"""Network validation: the single source of truth shared by every entry point.

Both the single-run API and the batch/concurrency scan go through
:func:`validate_network` against the exact same :class:`Network` object, so a
network can never validate in one path and fail in another.

Checks run in a fixed order so the reported cause is deterministic:

  0. ``invalid_initial_state``   - negative initial concentration
  1. ``missing_species``         - referenced species/enzyme not declared
  2. ``invalid_stoichiometry``   - empty or net-zero reaction
  3. ``invalid_rate``            - infeasible rate-law configuration
  4. ``unclosed_network``        - atom balance violated by a reaction
"""

from __future__ import annotations

from collections import defaultdict

from .errors import (
    InvalidInitialStateError,
    InvalidRateError,
    InvalidStoichiometryError,
    MissingSpeciesError,
    UnclosedNetworkError,
)
from .models import Network, RateLaw, RateLawKind


def _referenced_species(network: Network) -> set[str]:
    refs: set[str] = set()
    for rx in network.reactions:
        refs.update(rx.reactants)
        refs.update(rx.products)
        for rl in (rx.forward, rx.reverse):
            if rl is not None and rl.enzyme is not None:
                refs.add(rl.enzyme)
    return refs


def _check_initial_state(network: Network) -> None:
    for sp in network.species:
        if sp.initial_concentration < 0:
            raise InvalidInitialStateError(
                f"initial concentration of '{sp.id}' must be >= 0",
                details={"species": sp.id,
                         "initial_concentration": sp.initial_concentration},
            )


def _check_missing_species(network: Network) -> None:
    declared = {sp.id for sp in network.species}
    for missing in sorted(_referenced_species(network) - declared):
        raise MissingSpeciesError(
            f"reaction network references undeclared species '{missing}'",
            details={"species": missing, "declared": sorted(declared)},
        )


def _check_stoichiometry(network: Network) -> None:
    for rx in network.reactions:
        if not rx.reactants and not rx.products:
            raise InvalidStoichiometryError(
                f"reaction '{rx.id}' has neither reactants nor products",
                details={"reaction": rx.id},
            )
        net = defaultdict(float)
        for sp, c in rx.reactants.items():
            net[sp] -= c
        for sp, c in rx.products.items():
            net[sp] += c
        if all(abs(v) == 0.0 for v in net.values()):
            raise InvalidStoichiometryError(
                f"reaction '{rx.id}' has zero net stoichiometric change",
                details={"reaction": rx.id, "net": dict(net)},
            )


def _check_rates(network: Network) -> None:
    concentrations = {sp.id: sp.initial_concentration for sp in network.species}
    for rx in network.reactions:
        for tag, rl, side in (("forward", rx.forward, rx.reactants),
                              ("reverse", rx.reverse, rx.products)):
            if rl is None:
                continue
            _check_rate_law(rx.id, tag, rl, side, concentrations)


def _check_rate_law(reaction_id: str, tag: str, rl: RateLaw,
                    side: dict[str, float],
                    concentrations: dict[str, float]) -> None:
    ctx = f"reaction '{reaction_id}' {tag} rate"
    if rl.kind is RateLawKind.ENZYMATIC:
        # Michaelis-Menten describes one substrate converted by one enzyme.
        substrates = [sp for sp, c in side.items() if sp != rl.enzyme]
        if len(side) != 1 or not substrates:
            raise InvalidRateError(
                f"{ctx} is enzymatic but the {tag} side does not have exactly "
                "one substrate species (catalyst must not appear stoichiometrically)",
                details={"reaction": reaction_id, "direction": tag,
                         "side": sorted(side), "enzyme": rl.enzyme},
            )
        # An enzyme that is never present makes the rate law infeasible
        # (permanently zero) — rejected rather than silently accepted.
        if concentrations[rl.enzyme] <= 0.0:
            raise InvalidRateError(
                f"{ctx} uses enzyme '{rl.enzyme}' whose initial concentration "
                "is zero",
                details={"reaction": reaction_id, "direction": tag,
                         "enzyme": rl.enzyme},
            )


def _atom_composition(network: Network) -> dict[str, dict[str, float]]:
    """Effective atom composition per species.

    Species without an explicit ``composition`` are treated as a unit atom of
    their own name. This never silently "fixes" a network: a reaction that
    creates such a species out of nothing still fails the balance check.
    """
    comp: dict[str, dict[str, float]] = {}
    for sp in network.species:
        if sp.composition is not None:
            comp[sp.id] = dict(sp.composition)
        else:
            comp[sp.id] = {sp.id: 1.0}
    return comp


def _check_closure(network: Network) -> None:
    comp = _atom_composition(network)
    for rx in network.reactions:
        balance: dict[str, float] = defaultdict(float)
        for sp, c in rx.reactants.items():
            for atom, a in comp[sp].items():
                balance[atom] -= a * c
        for sp, c in rx.products.items():
            for atom, a in comp[sp].items():
                balance[atom] += a * c
        imbalance = {atom: v for atom, v in balance.items() if abs(v) > 0.0}
        if imbalance:
            raise UnclosedNetworkError(
                f"reaction '{rx.id}' is not atom-balanced; the network is not "
                "closed",
                details={"reaction": rx.id, "imbalance": imbalance},
            )


def validate_network(network: Network) -> None:
    """Raise a :class:`NetworkValidationError` subclass iff the network is invalid."""
    _check_initial_state(network)
    _check_missing_species(network)
    _check_stoichiometry(network)
    _check_rates(network)
    _check_closure(network)
