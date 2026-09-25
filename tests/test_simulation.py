"""Conservation, non-negativity and deterministic time-course checks."""

import logging

import numpy as np
import pytest

from app import simulation as sim_mod
from app.errors import ErrorCode, SimulationError
from app.models import Network, RateLaw, Reaction
from app.simulation import simulate

log = logging.getLogger("test.simulation")


def reversible_ab(kf=2.0, kr=1.0) -> Network:
    return Network(
        species={"A": 1.0, "B": 0.0},
        reactions=(
            Reaction(
                "r",
                {"A": 1},
                {"B": 1},
                RateLaw("mass_action", {"k": kf}),
                RateLaw("mass_action", {"k": kr}),
            ),
        ),
        version="ab-v1",
    )


def test_closed_network_conserves_total_mass():
    net = reversible_ab()
    tr = simulate(net, t_end=10.0, n_points=51)
    totals = tr.y.sum(axis=1)
    log.info("species=%s initial_total=1.0 totals(first,last)=%.12f,%.12f",
             tr.species, totals[0], totals[-1])
    assert np.allclose(totals, 1.0, atol=1e-9), totals
    assert tr.max_conservation_deviation < 1e-9
    log.info("max deviation %.3e within tol -> PASS", tr.max_conservation_deviation)


def test_steady_state_of_reversible_reaction():
    net = reversible_ab()
    tr = simulate(net, t_end=20.0, n_points=21)
    a_eq, b_eq = tr.y[-1]
    log.info("steady state [A]=%.6f [B]=%.6f (expect 1/3, 2/3)", a_eq, b_eq)
    assert abs(a_eq - 1 / 3) < 1e-6
    assert abs(b_eq - 2 / 3) < 1e-6


def test_finer_output_grid_does_not_change_trajectory():
    net = reversible_ab()
    coarse = simulate(net, t_end=10.0, n_points=11)
    fine = simulate(net, t_end=10.0, n_points=1001)
    # compare at the shared time points t = 0,1,...,10
    fine_at_coarse = fine.y[::100]
    log.info("coarse n_points=%d fine n_points=%d", 11, 1001)
    log.info("max trajectory difference at shared times: %.3e",
             np.max(np.abs(coarse.y - fine_at_coarse)))
    assert np.allclose(coarse.y, fine_at_coarse, atol=1e-10)
    assert np.allclose(coarse.y[-1], fine.y[-1], atol=1e-12)


def test_segmented_time_interval_matches_single_run():
    """[0,5] and [5,10] segments must reproduce the one-shot [0,10] run."""
    net = reversible_ab()
    full = simulate(net, t_end=10.0, n_points=21)
    seg1 = simulate(net, t_end=5.0, n_points=11)
    seg2 = simulate(net, t_end=10.0, t_start=5.0, n_points=11)
    log.info("full times %s", list(full.t))
    log.info("seg1 covers [0,5], seg2 covers [5,10]; t=5 shared")
    # seg1 matches full[0:11]; seg2 matches full[10:21] (t=5 is the shared seam)
    assert np.allclose(seg1.y, full.y[:11], atol=1e-10)
    assert np.allclose(seg2.y, full.y[10:], atol=1e-10)
    log.info("segments match single run at every shared time (atol=1e-10)")


def test_concentrations_never_negative():
    net = reversible_ab()
    tr = simulate(net, t_end=10.0, n_points=101)
    log.info("min observed concentration: %.3e", tr.min_concentration)
    assert tr.min_concentration >= 0.0
    assert np.all(tr.y >= 0.0)


def test_negative_concentration_is_detected(monkeypatch):
    """Inject a trajectory whose solver output dips negative; the invariant
    check must raise NEGATIVE_CONCENTRATION rather than hide it."""
    net = reversible_ab()
    t = np.linspace(0.0, 1.0, 2)
    y = np.array([[1.0, 0.0], [-2e-5, 1.00002]])  # A negative, total stays 1.0
    monkeypatch.setattr(sim_mod, "_integrate", lambda n, t_end: (t, y))
    log.info("injected trajectory=%s with negative entry %.1e", y.tolist(), -2e-5)
    with pytest.raises(SimulationError) as excinfo:
        simulate(net, t_end=1.0, n_points=2)
    log.info("raised code=%s detail=%s", excinfo.value.code, excinfo.value.detail)
    assert excinfo.value.code == ErrorCode.NEGATIVE_CONCENTRATION


def test_conservation_violation_is_detected(monkeypatch):
    """A drifting trajectory beyond tolerance must raise
    CONSERVATION_VIOLATION with locatable magnitudes."""
    net = reversible_ab()
    t = np.linspace(0.0, 1.0, 2)
    y = np.array([[1.0, 0.0], [0.5, 0.505]])  # total 1.0 -> 1.005
    monkeypatch.setattr(sim_mod, "_integrate", lambda n, t_end: (t, y))
    log.info("injected trajectory=%s totals=%s", y.tolist(), y.sum(axis=1).tolist())
    with pytest.raises(SimulationError) as excinfo:
        simulate(net, t_end=1.0, n_points=2, conservation_tol=1e-6)
    log.info("raised code=%s detail=%s", excinfo.value.code, excinfo.value.detail)
    assert excinfo.value.code == ErrorCode.CONSERVATION_VIOLATION


def test_tolerance_tighter_than_solver_noise_rejects():
    """Asking for tighter conservation than machine precision is reported,
    proving the tolerance is actually evaluated (not a rubber stamp)."""
    net = reversible_ab()
    with pytest.raises(SimulationError) as excinfo:
        simulate(net, t_end=10.0, n_points=11, conservation_tol=1e-18)
    log.info("tol=1e-18 -> %s: %s", excinfo.value.code, excinfo.value.detail)
    assert excinfo.value.code == ErrorCode.CONSERVATION_VIOLATION
