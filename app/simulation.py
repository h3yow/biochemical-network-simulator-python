"""Deterministic time-course simulation with invariant checking.

Determinism strategy
--------------------
The ODE is always integrated once from the network's initial state with
fixed solver settings (LSODA, rtol=1e-9, atol=1e-12) and dense output.
Requested output grids (``n_points``) and time sub-intervals only
*sample* the same dense solution, so refining the output step or
splitting an interval into segments can never change the trajectory or
the steady-state conclusion.

Caching
-------
Integrated solutions are cached under a SHA-256 fingerprint of the full
network description (species, reactions, rate parameters, boundary set,
version string). Identical requests are therefore reproducible, while a
different model version or any changed rate parameter yields a different
key and can never reuse a stale result. Cached arrays are read-only and
the cache is guarded by a lock, so concurrent requests neither share
mutable state nor pollute each other.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .errors import ErrorCode, SimulationError
from .models import Network
from .validation import validate_network

#: solver tolerances (fixed, independent of output grid)
_RTOL = 1e-9
_ATOL = 1e-12
#: a concentration more negative than this is non-physical
_NEG_TOL = 1e-9

_cache_lock = threading.Lock()
_solution_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class ConservationRecord:
    """One independent conserved quantity of the network and its drift.

    ``name`` is the element/group for composition-annotated networks, or
    ``"total"`` (sum of free-species concentrations) for legacy ones.
    """

    name: str
    initial: float
    max_deviation: float
    limit: float
    passed: bool


@dataclass(frozen=True)
class Trajectory:
    """Simulation result: time points and concentrations."""

    t: np.ndarray
    y: np.ndarray  # shape (n_times, n_species)
    species: tuple[str, ...]
    conserved_total: float
    max_conservation_deviation: float
    min_concentration: float
    conservations: tuple[ConservationRecord, ...] = ()


def network_fingerprint(network: Network) -> str:
    """Canonical fingerprint covering every field that affects results.

    Compositions enter the key only when declared, so legacy networks
    without compositions keep their historical fingerprints bit-for-bit,
    while two networks with identical species names but different
    compositions never share cached results.
    """
    payload = {
        "version": network.version,
        "species": sorted(network.species.items()),
        "boundary": sorted(network.boundary_species),
        "reactions": [
            {
                "name": r.name,
                "reactants": sorted(r.reactants.items()),
                "products": sorted(r.products.items()),
                "rate": [r.rate.kind, sorted(r.rate.params.items())],
                "reverse": None
                if r.reverse is None
                else [r.reverse.kind, sorted(r.reverse.params.items())],
            }
            for r in network.reactions
        ],
    }
    if network.compositions:
        payload["compositions"] = sorted(
            (name, sorted(comp.items()))
            for name, comp in network.compositions.items()
        )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _build_rhs(network: Network, species: tuple[str, ...]):
    index = {name: i for i, name in enumerate(species)}
    boundary_idx = {index[name] for name in network.boundary_species}

    # stoichiometry: list of (rate_fn, delta_vector)
    terms = []
    for reaction in network.reactions:
        delta = np.zeros(len(species))
        for name, coeff in reaction.reactants.items():
            delta[index[name]] -= coeff
        for name, coeff in reaction.products.items():
            delta[index[name]] += coeff

        reactant_idx = [index[n] for n in reaction.reactants]
        reactant_coeff = list(reaction.reactants.values())

        def make_rate(law, ridx=reactant_idx, rcoeff=reactant_coeff):
            if law.kind == "mass_action":
                k = law.params["k"]

                def rate(y):
                    v = k
                    for i, c in zip(ridx, rcoeff):
                        v *= max(y[i], 0.0) ** c
                    return v

                return rate
            # michaelis_menten: single substrate, enzyme by name
            kcat = law.params["kcat"]
            km = law.params["Km"]
            eidx = index[law.params["enzyme"]]
            sidx = ridx[0]

            def rate(y):
                s = max(y[sidx], 0.0)
                return kcat * max(y[eidx], 0.0) * s / (km + s)

            return rate

        # the reverse law consumes the *products* of the forward reaction
        product_idx = [index[n] for n in reaction.products]
        product_coeff = list(reaction.products.values())
        fwd = make_rate(reaction.rate)
        rev = (
            make_rate(reaction.reverse, product_idx, product_coeff)
            if reaction.reverse is not None
            else None
        )
        terms.append((fwd, rev, delta))

    def rhs(_t, y):
        dydt = np.zeros(len(species))
        for fwd, rev, delta in terms:
            v = fwd(y)
            if rev is not None:
                v -= rev(y)
            dydt += v * delta
        for i in boundary_idx:
            dydt[i] = 0.0
        return dydt

    return rhs


def _integrate(network: Network, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Integrate once from t=0; return (t_grid, y_grid) on a fine grid."""
    key = network_fingerprint(network) + f"@{t_end!r}"
    with _cache_lock:
        hit = _solution_cache.get(key)
    if hit is not None:
        return hit

    species = tuple(sorted(network.species))
    y0 = np.array([network.species[s] for s in species], dtype=float)
    rhs = _build_rhs(network, species)
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        method="LSODA",
        rtol=_RTOL,
        atol=_ATOL,
        dense_output=True,
    )
    if not sol.success:
        raise SimulationError(
            ErrorCode.INFEASIBLE_RATE, f"ODE solver failed: {sol.message}"
        )
    # sample the dense solution on a fine deterministic grid
    t_grid = np.linspace(0.0, t_end, max(2, int(t_end / 1e-3) + 1))
    y_grid = sol.sol(t_grid).T
    y_grid.setflags(write=False)
    t_grid.setflags(write=False)
    with _cache_lock:
        _solution_cache[key] = (t_grid, y_grid)
    return t_grid, y_grid


def simulate(
    network: Network,
    t_end: float,
    n_points: int = 101,
    t_start: float = 0.0,
    conservation_tol: float = 1e-6,
) -> Trajectory:
    """Simulate the network deterministically on [t_start, t_end].

    Enforces: every conserved quantity of a closed network stays within
    ``conservation_tol`` (one quantity per declared element/group when
    compositions are present, otherwise the legacy total concentration);
    no non-physical negative concentrations. Raises SimulationError with
    a distinguishing code naming the violated quantity otherwise.
    """
    validate_network(network)
    if t_end <= t_start or t_start < 0:
        raise SimulationError(
            ErrorCode.INFEASIBLE_RATE,
            f"invalid time interval [{t_start}, {t_end}]",
        )
    species = tuple(sorted(network.species))
    t_fine, y_fine = _integrate(network, t_end)

    # sample the fine trajectory at the requested output grid
    t_out = np.linspace(t_start, t_end, n_points)
    y_out = np.empty((n_points, len(species)))
    for j in range(len(species)):
        y_out[:, j] = np.interp(t_out, t_fine, y_fine[:, j])

    # --- invariant checks on the fine (unsampled) trajectory ---
    free_idx = [i for i, s in enumerate(species) if s not in network.boundary_species]
    free_names = [species[i] for i in free_idx]
    y_free = y_fine[:, free_idx]

    # conserved quantities: one per declared element/group when the network
    # carries compositions, otherwise the legacy single "total" (sum of
    # free-species concentrations) so old verdicts are unchanged
    if network.compositions:
        quantities = [
            (element, np.array([network.compositions[s].get(element, 0.0)
                                for s in free_names]))
            for element in sorted({e for c in network.compositions.values()
                                   for e in c})
        ]
    else:
        quantities = [("total", np.ones(len(free_names)))]

    records = []
    for name, weights in quantities:
        totals = y_free @ weights
        initial = float(totals[0])
        max_dev = float(np.max(np.abs(totals - initial)))
        limit = conservation_tol * max(abs(initial), 1.0)
        records.append(
            ConservationRecord(
                name=name,
                initial=initial,
                max_deviation=max_dev,
                limit=limit,
                passed=max_dev <= limit,
            )
        )
    conserved_total = records[0].initial
    max_dev = max(r.max_deviation for r in records)
    if network.closed:
        failures = [r for r in records if not r.passed]
        if failures:
            summary = "; ".join(
                f"'{r.name}' deviates by {r.max_deviation:.3e} "
                f"(tolerance {r.limit:.3e})" for r in failures
            )
            raise SimulationError(
                ErrorCode.CONSERVATION_VIOLATION,
                f"conserved quantity {summary} over [0, {t_end}]",
            )
    min_conc = float(np.min(y_free)) if free_idx else 0.0
    if min_conc < -_NEG_TOL:
        raise SimulationError(
            ErrorCode.NEGATIVE_CONCENTRATION,
            f"concentration reached {min_conc:.3e} (below -{_NEG_TOL:.0e})",
        )
    # clamp solver-level noise within tolerance to exactly zero
    y_out = np.clip(y_out, 0.0, None)
    return Trajectory(
        t=t_out,
        y=y_out,
        species=species,
        conserved_total=conserved_total,
        max_conservation_deviation=max_dev,
        min_concentration=max(min_conc, 0.0) if min_conc > -_NEG_TOL else min_conc,
        conservations=tuple(records),
    )
