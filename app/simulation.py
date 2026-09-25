"""Deterministic time-course integration and conservation checks.

Numerical scheme
----------------
* Classical fixed-step RK4, integrated independently over each output
  interval ``[t_j, t_{j+1}]``. Rates have no explicit time dependence, so
  partitioning the same interval into segments reproduces the identical
  floating-point trajectory.
* The internal sub-step count per output interval is chosen by Richardson
  refinement (compare ``n`` vs ``2n`` RK4 sub-steps). Shrinking the output
  step therefore only *re-samples* an already-converged trajectory; it never
  changes steady-state or trajectory conclusions.
* Failure to reach the configured accuracy raises :class:`IntegrationError`
  rather than silently returning an inaccurate result.
"""

from __future__ import annotations

import numpy as np

from .errors import (
    ConservationViolationError,
    IntegrationError,
    NonPhysicalResultError,
)
from .models import Network, RateLaw, RateLawKind, SimConfig, SimResult

# Richardson convergence on RK4: error estimate magnitude below which the
# internal step is accepted, plus a hard cap on halvings.
_MAX_REFINEMENTS = 12
_STEADY_STATE_RATE_TOL = 1e-8


# ---------------------------------------------------------------------------
# Structural matrices
# ---------------------------------------------------------------------------

def _index_maps(network: Network) -> tuple[list[str], dict[str, int]]:
    order = [sp.id for sp in network.species]
    return order, {sid: i for i, sid in enumerate(order)}


def build_stoichiometry(network: Network) -> tuple[np.ndarray, list[str]]:
    """Return ``(N, directed_reaction_ids)``.

    A reversible reaction expands into two directed columns (forward, reverse).
    """
    order, idx = _index_maps(network)
    n_sp = len(order)
    directed: list[str] = []
    columns: list[np.ndarray] = []
    for rx in network.reactions:
        col = np.zeros(n_sp)
        for sp, c in rx.reactants.items():
            col[idx[sp]] -= c
        for sp, c in rx.products.items():
            col[idx[sp]] += c
        directed.append(rx.id)
        columns.append(col)
        if rx.reversible:
            directed.append(rx.id + "<-")
            columns.append(-col)
    return np.stack(columns, axis=1), directed


def _law_rate(law: RateLaw, substrate_id: str, y: np.ndarray,
              idx: dict[str, int]) -> float:
    if law.kind is RateLawKind.MASS_ACTION:
        return law.k
    e = max(y[idx[law.enzyme]], 0.0)
    s = max(y[idx[substrate_id]], 0.0)
    return law.k * e * s / (law.km + s)


def reaction_rates(network: Network, y: np.ndarray) -> np.ndarray:
    """Evaluate directed-reaction rates at concentration vector ``y``."""
    _, idx = _index_maps(network)
    n_dir = len(network.reactions) + sum(
        1 for r in network.reactions if r.reversible)
    rates = np.zeros(n_dir)
    k = 0
    for rx in network.reactions:
        # forward direction: mass-action over reactants, MM over sole reactant
        law = rx.forward
        if law.kind is RateLawKind.MASS_ACTION:
            v = law.k
            for sp, c in rx.reactants.items():
                v *= max(y[idx[sp]], 0.0) ** c
        else:
            v = _law_rate(law, next(iter(rx.reactants)), y, idx)
        rates[k] = v
        k += 1
        if rx.reversible:
            law = rx.reverse
            if law.kind is RateLawKind.MASS_ACTION:
                v = law.k
                for sp, c in rx.products.items():
                    v *= max(y[idx[sp]], 0.0) ** c
            else:
                v = _law_rate(law, next(iter(rx.products)), y, idx)
            rates[k] = v
            k += 1
    return rates


def _rhs(network: Network, stoich: np.ndarray) -> "callable":
    def f(y: np.ndarray) -> np.ndarray:
        return stoich @ reaction_rates(network, y)
    return f


# ---------------------------------------------------------------------------
# RK4 with Richardson step selection
# ---------------------------------------------------------------------------

def _rk4_advance(f, y0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    y = y0
    h = dt / n_steps
    for _ in range(n_steps):
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _integrate_interval(f, y0: np.ndarray, dt: float,
                        rel_tol: float, abs_tol: float) -> np.ndarray:
    """Integrate one output interval with adaptive Richardson refinement."""
    n = 1
    prev = _rk4_advance(f, y0, dt, n)
    for _ in range(_MAX_REFINEMENTS):
        n2 = n * 2
        nxt = _rk4_advance(f, y0, dt, n2)
        scale = np.maximum(np.abs(prev), np.abs(nxt))
        tol = abs_tol + rel_tol * scale
        if np.all(np.abs(nxt - prev) <= tol):
            return nxt
        prev, n = nxt, n2
    raise IntegrationError(
        "RK4 Richardson refinement failed to reach requested tolerance",
        details={"interval_length": dt, "sub_steps": n * 2,
                 "rel_tol": rel_tol, "abs_tol": abs_tol,
                 "max_error": float(np.max(np.abs(nxt - prev)))},
    )


# ---------------------------------------------------------------------------
# Conservation / physicality
# ---------------------------------------------------------------------------

def conservation_matrix(network: Network) -> np.ndarray:
    """Rows are atom-conservation vectors (one per distinct atom type)."""
    atoms: list[str] = []
    for sp in network.species:
        comp = sp.composition or {sp.id: 1.0}
        for a in comp:
            if a not in atoms:
                atoms.append(a)
    a_index = {a: i for i, a in enumerate(atoms)}
    mat = np.zeros((len(atoms), len(network.species)))
    for j, sp in enumerate(network.species):
        comp = sp.composition or {sp.id: 1.0}
        for a, amount in comp.items():
            mat[a_index[a], j] = amount
    return mat


def _check_state(y: np.ndarray, cfg: SimConfig, t: float) -> np.ndarray:
    worst = float(np.min(y))
    if worst < -cfg.negative_conc_tol:
        viol = int(np.argmin(y))
        raise NonPhysicalResultError(
            f"non-physical negative concentration {worst:.3e} at t={t:g}",
            details={"time": t, "species_index": viol, "concentration": worst},
        )
    # Numerical noise within tolerance is clamped; real violations are raised.
    return np.where(y < 0.0, 0.0, y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(network: Network, cfg: SimConfig,
             breakpoints: list[float] | None = None) -> SimResult:
    """Run a deterministic time course over ``[0, t_end]``.

    ``breakpoints`` partitions the interval (each segment re-enters through
    :func:`simulate_interval`, exactly the way a chunked client would); the
    trajectory at output points is identical with or without breakpoints.
    """
    if cfg.t_end <= 0 or cfg.output_step <= 0:
        raise IntegrationError(
            "t_end and output_step must be > 0",
            details={"t_end": cfg.t_end, "output_step": cfg.output_step},
        )
    order = [sp.id for sp in network.species]
    y0 = np.array([sp.initial_concentration for sp in network.species],
                  dtype=float)
    a_mat = conservation_matrix(network)
    totals0 = a_mat @ y0
    conserved_total = float(np.sum(totals0))

    # Global output grid; times are always the same index*h regardless of how
    # the interval is partitioned, so chunked and single runs are identical.
    n_total = max(1, int(round(cfg.t_end / cfg.output_step)))
    h = cfg.t_end / n_total
    cut_idx = [0, n_total]
    for b in breakpoints or []:
        k = int(round(b / h))
        if not (0 < k < n_total) or abs(b - k * h) > 1e-9 * max(1.0, abs(b)):
            raise IntegrationError(
                f"breakpoint {b:g} does not align with the output grid "
                f"(h={h:g})",
                details={"breakpoint": b, "output_step": h},
            )
        cut_idx.append(k)
    cut_idx = sorted(set(cut_idx))

    stoich, _directed = build_stoichiometry(network)
    f = _rhs(network, stoich)

    times: list[float] = [0.0]
    rows: list[np.ndarray] = [y0.copy()]
    y = y0
    max_drift = 0.0
    min_conc = float(np.min(y0))

    for a, b in zip(cut_idx[:-1], cut_idx[1:]):
        for j in range(a + 1, b + 1):
            y = _integrate_interval(f, y, h, cfg.rel_tol, cfg.abs_tol)
            t = j * h
            y = _check_state(y, cfg, t)
            rel_drift = np.abs(a_mat @ y - totals0) / np.maximum(np.abs(totals0), 1.0)
            max_drift = max(max_drift, float(np.max(rel_drift)))
            times.append(t)
            rows.append(y.copy())
            min_conc = min(min_conc, float(np.min(y)))

    if max_drift > cfg.conservation_tol:
        raise ConservationViolationError(
            f"conserved mass drifted by {max_drift:.3e} > "
            f"tolerance {cfg.conservation_tol:g}",
            details={"max_relative_error": max_drift,
                     "tolerance": cfg.conservation_tol},
        )

    dydt = f(y)
    scale = np.maximum(np.abs(y), 1.0)
    steady: dict[str, float] | None = None
    if np.max(np.abs(dydt) / scale) <= _STEADY_STATE_RATE_TOL:
        steady = {sid: float(v) for sid, v in zip(order, y)}

    return SimResult(
        model_id=network.model_id,
        version=network.version,
        times=tuple(float(t) for t in times),
        species_order=tuple(order),
        values=tuple(tuple(float(v) for v in row) for row in rows),
        conserved_total=conserved_total,
        max_conservation_error=max_drift,
        min_concentration=min_conc,
        steady_state=steady,
    )
