"""Conservation, physicality, step-size / chunking consistency, steady state."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from app.errors import (
    ConservationViolationError,
    NonPhysicalResultError,
)
from app.models import SimConfig
from app.parser import parse_network, parse_sim_config
from app.simulation import simulate
from app.validation import validate_network

log = logging.getLogger("bionet.tests")


def _sim(payload: dict, **cfg_kw):
    network = parse_network(payload)
    validate_network(network)
    cfg = SimConfig(t_end=cfg_kw.pop("t_end", 30.0),
                    output_step=cfg_kw.pop("output_step", 1.0), **cfg_kw)
    return network, cfg, simulate(network, cfg)


def test_closed_mass_conserved(service):
    payload = {
        "model_id": "ab", "version": 1,
        "species": [
            {"id": "A", "initial_concentration": 2.0, "composition": {"X": 1}},
            {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
             "reversible": True,
             "forward": {"kind": "mass_action", "k": 1.0},
             "reverse": {"kind": "mass_action", "k": 3.0}},
        ],
    }
    _, cfg, result = _sim(payload)
    log.info("conservation: total=%g max_err=%.3e tol=%g final=%s",
             result.conserved_total, result.max_conservation_error,
             cfg.conservation_tol, result.steady_state)
    assert result.max_conservation_error <= cfg.conservation_tol
    # equilibrium: B/A = kf/kr = 1/3, total = 2
    assert abs(result.steady_state["A"] - 1.5) < 1e-6
    assert abs(result.steady_state["B"] - 0.5) < 1e-6
    assert result.min_concentration >= 0.0


def test_multiatom_conservation_and_enzymatic(service):
    payload = {
        "model_id": "mm", "version": 1,
        "species": [
            {"id": "S", "initial_concentration": 4.0,
             "composition": {"C": 6, "H": 12, "O": 6}},
            {"id": "P", "initial_concentration": 0.0,
             "composition": {"C": 6, "H": 12, "O": 6}},
            {"id": "E", "initial_concentration": 0.2},
        ],
        "reactions": [
            {"id": "cat", "reactants": {"S": 1}, "products": {"P": 1},
             "forward": {"kind": "enzymatic", "k": 5.0,
                         "enzyme": "E", "km": 0.5}}
        ],
    }
    _, cfg, result = _sim(payload, t_end=40.0)
    log.info("enzymatic: S_final=%.6f P_final=%.6f drift=%.3e",
             result.steady_state["S"], result.steady_state["P"],
             result.max_conservation_error)
    assert result.max_conservation_error <= cfg.conservation_tol
    assert result.steady_state["S"] < 1e-6
    assert abs(result.steady_state["P"] - 4.0) < 1e-6
    # enzyme (no composition) stays constant: it is catalyst-only
    s_idx = result.species_order.index("E")
    enzyme_trace = [row[s_idx] for row in result.values]
    assert max(abs(e - 0.2) for e in enzyme_trace) < 1e-9


def test_smaller_output_step_does_not_change_conclusion(service):
    payload = {
        "model_id": "ab", "version": 1,
        "species": [
            {"id": "A", "initial_concentration": 3.0, "composition": {"X": 1}},
            {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
             "forward": {"kind": "mass_action", "k": 0.7}}
        ],
    }
    network = parse_network(payload)
    validate_network(network)
    coarse = simulate(network, SimConfig(t_end=40.0, output_step=2.0))
    fine = simulate(network, SimConfig(t_end=40.0, output_step=0.25))
    # compare at shared output times
    coarse_at = {round(t, 9): v for t, v in zip(coarse.times, coarse.values)}
    diffs = []
    for t, row in zip(fine.times, fine.values):
        key = round(t, 9)
        if key in coarse_at:
            diffs.append(max(abs(a - b) for a, b in zip(row, coarse_at[key])))
    log.info("step-size consistency: shared points=%d max diff=%.3e",
             len(diffs), max(diffs))
    assert max(diffs) < 1e-8
    assert coarse.steady_state is not None and fine.steady_state is not None


def test_segmented_interval_matches_single_run(service):
    payload = {
        "model_id": "ab", "version": 1,
        "species": [
            {"id": "A", "initial_concentration": 1.0, "composition": {"X": 2}},
            {"id": "B", "initial_concentration": 1.0, "composition": {"X": 2}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
             "reversible": True,
             "forward": {"kind": "mass_action", "k": 2.0},
             "reverse": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    network = parse_network(payload)
    validate_network(network)
    cfg = SimConfig(t_end=24.0, output_step=0.5)
    single = simulate(network, cfg)
    chunked = simulate(network, cfg, breakpoints=[3.0, 7.5, 13.0, 19.0])
    assert single.times == chunked.times
    diffs = [abs(a - b) for r1, r2 in zip(single.values, chunked.values)
             for a, b in zip(r1, r2)]
    log.info("segmented consistency: points=%d max diff=%.3e",
             len(single.times), max(diffs))
    assert max(diffs) == 0.0  # identical integration from identical states
    assert single.steady_state == chunked.steady_state or (
        abs(single.steady_state["A"] - chunked.steady_state["A"]) < 1e-12
    )


def test_conservation_violation_is_reported_for_source_network():
    """Runtime conservation check: a net source violates atom conservation.

    Closure validation rejects such networks at registration; here the domain
    object is built directly (bypassing the parser) to exercise the runtime
    guard itself and prove it raises with locatable details.
    """
    from dataclasses import replace
    payload = {
        "model_id": "src", "version": 1,
        "species": [
            {"id": "A", "initial_concentration": 1.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "in", "reactants": {"A": 1}, "products": {"A": 1},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    network = parse_network(payload)
    bad_rx = replace(network.reactions[0], products={"A": 2.0})  # net +A source
    bad = replace(network, reactions=(bad_rx,))
    with pytest.raises(ConservationViolationError) as ei:
        simulate(bad, SimConfig(t_end=5.0, output_step=1.0,
                                conservation_tol=1e-8))
    log.info("conservation failure caught: %s details=%s",
             ei.value.message, ei.value.details)
    assert ei.value.details["max_relative_error"] > 1e-8


def test_non_physical_negative_concentration_detected():
    """Zero-order sink (rate k, net stoich -A) drives A below zero.

    The parser/validator never allows such a description; the object is built
    directly to exercise the integration-level non-physicality guard.
    """
    from dataclasses import replace
    payload = {
        "model_id": "sink", "version": 1,
        "species": [
            {"id": "A", "initial_concentration": 0.1, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "out", "reactants": {"A": 1}, "products": {"A": 1},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    network = parse_network(payload)
    # no reactant side -> zero-order rate k; negative product coefficient is a
    # net sink (parser would reject, constructed here on purpose)
    bad_rx = replace(network.reactions[0], reactants={}, products={"A": -1.0})
    bad = replace(network, reactions=(bad_rx,))
    with pytest.raises(NonPhysicalResultError) as ei:
        simulate(bad, SimConfig(t_end=2.0, output_step=0.5))
    log.info("negative concentration caught: %s details=%s",
             ei.value.message, ei.value.details)
    assert ei.value.details["concentration"] < 0.0
    assert "time" in ei.value.details


def test_tiny_negative_numerical_noise_is_clamped():
    from app.models import SimConfig as C
    from app.simulation import _check_state
    cfg = C(t_end=1.0, output_step=1.0)
    y = np.array([-1e-12, 1.0])
    fixed = _check_state(y, cfg, t=0.3)
    assert fixed[0] == 0.0
    with pytest.raises(NonPhysicalResultError):
        _check_state(np.array([-1e-6, 1.0]), cfg, t=0.3)


def test_infeasible_config_rejected():
    from app.errors import SimConfigError
    with pytest.raises(SimConfigError):
        parse_sim_config({"t_end": 0, "output_step": 1})
    with pytest.raises(SimConfigError):
        parse_sim_config({"t_end": 1.0, "output_step": 1.0, "rel_tol": 0})
