"""Composition-aware networks: element balance validation, multiple
independent conserved quantities, legacy compatibility and per-point
scan failure isolation."""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import simulation as sim_mod
from app.errors import ErrorCode, NetworkValidationError, SimulationError
from app.models import Network, RateLaw, Reaction
from app.scanning import perturb_parameter, scan_parameter_isolated
from app.service import create_app
from app.simulation import network_fingerprint, simulate
from app.validation import validate_network

log = logging.getLogger("test.composition")

MA = lambda k: RateLaw("mass_action", {"k": k})  # noqa: E731


def synthesis_network(kf=0.5, kr=0.1) -> Network:
    """Balanced multi-substrate network 2A + B <-> C with compositions."""
    return Network(
        species={"A": 2.0, "B": 1.0, "C": 0.0},
        reactions=(
            Reaction("synth", {"A": 2, "B": 1}, {"C": 1}, MA(kf), MA(kr)),
        ),
        compositions={
            "A": {"C": 1, "H": 1},
            "B": {"C": 1},
            "C": {"C": 3, "H": 2},
        },
        version="synth-v1",
    )


def split_network() -> Network:
    """Balanced cleavage A -> B + C with non-1:1 stoichiometry."""
    return Network(
        species={"A": 1.0, "B": 0.0, "C": 0.0},
        reactions=(
            Reaction("cleave", {"A": 1}, {"B": 2, "C": 1}, MA(0.8), MA(0.05)),
        ),
        compositions={
            "A": {"C": 4, "H": 2},
            "B": {"C": 1, "H": 1},
            "C": {"C": 2},
        },
        version="split-v1",
    )


# ---------------------------------------------------------------- import + balance

def test_multi_substrate_balanced_network_validates_and_simulates():
    net = synthesis_network()
    validate_network(net)
    tr = simulate(net, t_end=10.0, n_points=21)
    names = [r.name for r in tr.conservations]
    log.info("network 2A+B<->C conserved quantities: %s", names)
    for r in tr.conservations:
        log.info("  %s initial=%.6f max_dev=%.3e limit=%.3e passed=%s",
                 r.name, r.initial, r.max_deviation, r.limit, r.passed)
    assert names == ["C", "H"]  # two independent conserved quantities
    assert all(r.passed for r in tr.conservations)
    assert all(r.max_deviation < 1e-9 for r in tr.conservations)


def test_split_reaction_with_non_unit_stoichiometry_validates():
    net = split_network()
    validate_network(net)
    tr = simulate(net, t_end=10.0, n_points=21)
    log.info("cleavage A->2B+C conservations: %s",
             [(r.name, round(r.max_deviation, 12)) for r in tr.conservations])
    assert [r.name for r in tr.conservations] == ["C", "H"]
    assert all(r.passed for r in tr.conservations)


def test_unbalanced_reaction_points_to_reaction_and_element():
    net = Network(
        species={"A": 2.0, "B": 1.0, "C": 0.0},
        reactions=(Reaction("synth", {"A": 2, "B": 1}, {"C": 1}, MA(0.5)),),
        compositions={"A": {"C": 1}, "B": {"C": 1}, "C": {"C": 2}},  # 3 != 2
    )
    log.info("input: 2A+B->C with C-counts A=1,B=1,C=2 (reactants 3, products 2)")
    with pytest.raises(NetworkValidationError) as excinfo:
        validate_network(net)
    log.info("got code=%s detail=%s", excinfo.value.code, excinfo.value.detail)
    assert excinfo.value.code == ErrorCode.UNBALANCED_REACTION
    assert "synth" in excinfo.value.detail
    assert "'C'" in excinfo.value.detail


def test_reverse_direction_imbalance_is_caught():
    """Forward and reverse share stoichiometry: an imbalanced reversible
    reaction is rejected by the same single check."""
    net = Network(
        species={"A": 1.0, "B": 0.0},
        reactions=(Reaction("r", {"A": 1}, {"B": 1}, MA(1.0), MA(0.5)),),
        compositions={"A": {"C": 2}, "B": {"C": 1}},
    )
    with pytest.raises(NetworkValidationError) as excinfo:
        validate_network(net)
    log.info("reversible imbalanced -> %s: %s",
             excinfo.value.code, excinfo.value.detail)
    assert excinfo.value.code == ErrorCode.UNBALANCED_REACTION


# ------------------------------------------------------------- invalid compositions

INVALID_COMP_CASES = {
    "negative_count": ({"A": {"C": -1}, "B": {"C": 1}},
                       ErrorCode.INVALID_COMPOSITION),
    "zero_count": ({"A": {"C": 0}, "B": {"C": 1}},
                   ErrorCode.INVALID_COMPOSITION),
    "nan_count": ({"A": {"C": float("nan")}, "B": {"C": 1}},
                  ErrorCode.INVALID_COMPOSITION),
    "empty_composition": ({"A": {}, "B": {"C": 1}},
                          ErrorCode.INVALID_COMPOSITION),
    "partial_declaration": ({"A": {"C": 1}},  # B missing -> contradictory
                            ErrorCode.INVALID_COMPOSITION),
    "unknown_species": ({"A": {"C": 1}, "B": {"C": 1}, "GHOST": {"C": 1}},
                        ErrorCode.MISSING_SPECIES),
}


@pytest.mark.parametrize("name", sorted(INVALID_COMP_CASES))
def test_invalid_compositions_raise_distinct_codes(name):
    compositions, expected = INVALID_COMP_CASES[name]
    net = Network(
        species={"A": 1.0, "B": 0.0},
        reactions=(Reaction("r", {"A": 1}, {"B": 1}, MA(1.0)),),
        compositions=compositions,
    )
    log.info("case=%s compositions=%s", name, compositions)
    with pytest.raises(NetworkValidationError) as excinfo:
        validate_network(net)
    log.info("got code=%s detail=%s", excinfo.value.code, excinfo.value.detail)
    assert excinfo.value.code == expected


def test_api_and_offline_composition_validation_agree():
    net = {
        "species": {"A": 2.0, "B": 1.0, "C": 0.0},
        "reactions": [{"name": "synth", "reactants": {"A": 2, "B": 1},
                       "products": {"C": 1},
                       "rate": {"kind": "mass_action", "params": {"k": 0.5}}}],
        "compositions": {"A": {"C": 1}, "B": {"C": 1}, "C": {"C": 2}},
    }
    client = TestClient(create_app())
    resp = client.post("/validate", json=net)
    log.info("http verdict: %s %s", resp.status_code, resp.json())
    assert resp.status_code == 422
    assert resp.json()["code"] == "UNBALANCED_REACTION"
    assert "synth" in resp.json()["detail"] and "'C'" in resp.json()["detail"]


# --------------------------------------------- multiple independent conserved quantities

def two_pool_network() -> Network:
    """Two independent pools: A<->B conserves element C, X<->Y conserves H."""
    return Network(
        species={"A": 1.0, "B": 0.0, "X": 0.5, "Y": 0.25},
        reactions=(
            Reaction("ab", {"A": 1}, {"B": 1}, MA(1.0), MA(0.5)),
            Reaction("xy", {"X": 1}, {"Y": 1}, MA(0.8), MA(0.2)),
        ),
        compositions={
            "A": {"C": 1}, "B": {"C": 1},
            "X": {"H": 1}, "Y": {"H": 1},
        },
        version="pools-v1",
    )


def test_independent_conservation_quantities_reported_separately():
    net = two_pool_network()
    tr = simulate(net, t_end=10.0, n_points=21)
    by_name = {r.name: r for r in tr.conservations}
    log.info("quantities: %s", {k: (v.initial, v.max_deviation)
                                for k, v in by_name.items()})
    assert set(by_name) == {"C", "H"}
    assert by_name["C"].initial == pytest.approx(1.0)
    assert by_name["H"].initial == pytest.approx(0.75)
    assert all(r.passed for r in tr.conservations)


def test_conservation_violation_names_the_failing_quantity(monkeypatch):
    """Inject a trajectory where only the C pool drifts; the error must
    name 'C' and the H record must still report as passed."""
    net = two_pool_network()
    t = np.linspace(0.0, 1.0, 2)
    # species order sorted: A, B, X, Y ; C pool 1.0 -> 1.1, H pool steady
    y = np.array([[1.0, 0.0, 0.5, 0.25],
                  [0.6, 0.5, 0.5, 0.25]])
    monkeypatch.setattr(sim_mod, "_integrate", lambda n, t_end: (t, y))
    log.info("injected trajectory=%s", y.tolist())
    with pytest.raises(SimulationError) as excinfo:
        simulate(net, t_end=1.0, n_points=2, conservation_tol=1e-6)
    log.info("raised code=%s detail=%s", excinfo.value.code, excinfo.value.detail)
    assert excinfo.value.code == ErrorCode.CONSERVATION_VIOLATION
    assert "'C'" in excinfo.value.detail
    assert "'H'" not in excinfo.value.detail


def test_conservation_verdict_stable_across_grids_and_segments():
    """Finer output grids and segmented evaluation must not change the
    conservation conclusion for composition networks."""
    net = synthesis_network()
    coarse = simulate(net, t_end=10.0, n_points=21)
    fine = simulate(net, t_end=10.0, n_points=2001)
    seg1 = simulate(net, t_end=5.0, n_points=11)
    seg2 = simulate(net, t_end=10.0, t_start=5.0, n_points=11)
    log.info("coarse devs=%s fine devs=%s",
             [r.max_deviation for r in coarse.conservations],
             [r.max_deviation for r in fine.conservations])
    assert np.allclose(coarse.y, fine.y[::100], atol=1e-10)
    assert np.allclose(seg1.y, coarse.y[:11], atol=1e-10)
    assert np.allclose(seg2.y, coarse.y[10:], atol=1e-10)
    for tr in (coarse, fine, seg1, seg2):
        assert all(r.passed for r in tr.conservations)


# ------------------------------------------------------------- legacy compatibility

def legacy_network() -> Network:
    return Network(
        species={"A": 1.0, "B": 0.0},
        reactions=(Reaction("r", {"A": 1}, {"B": 1}, MA(1.0), MA(0.5)),),
        version="demo-1",
    )


def test_legacy_fingerprint_bit_identical_without_compositions():
    """Networks without compositions keep their historical fingerprint."""
    net = legacy_network()
    payload = {
        "version": "demo-1",
        "species": [["A", 1.0], ["B", 0.0]],
        "boundary": [],
        "reactions": [{
            "name": "r",
            "reactants": [["A", 1]],
            "products": [["B", 1]],
            "rate": ["mass_action", [["k", 1.0]]],
            "reverse": ["mass_action", [["k", 0.5]]],
        }],
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    log.info("legacy fingerprint=%s expected=%s",
             network_fingerprint(net)[:16], expected[:16])
    assert network_fingerprint(net) == expected


def test_legacy_network_reports_single_total_quantity():
    net = legacy_network()
    tr = simulate(net, t_end=5.0, n_points=11)
    log.info("legacy conservations: %s",
             [(r.name, r.max_deviation) for r in tr.conservations])
    assert [r.name for r in tr.conservations] == ["total"]
    assert tr.conservations[0].initial == pytest.approx(1.0)
    assert tr.conserved_total == pytest.approx(1.0)


def test_same_species_names_different_compositions_never_share_results():
    base = synthesis_network()
    other = Network(
        species=base.species,
        reactions=base.reactions,
        version=base.version,
        compositions={"A": {"C": 1, "H": 2},  # different H count, still
                      "B": {"C": 1},          # balanced: C has H:4? no ->
                      "C": {"C": 3, "H": 4}}, # 2*2+0=4 vs 4 balanced
    )
    validate_network(other)
    fp_base, fp_other = network_fingerprint(base), network_fingerprint(other)
    log.info("fingerprints: %s vs %s", fp_base[:12], fp_other[:12])
    assert fp_base != fp_other
    # both simulate independently without cache confusion
    tr_a = simulate(base, t_end=5.0, n_points=11)
    tr_b = simulate(other, t_end=5.0, n_points=11)
    assert np.array_equal(tr_a.y, simulate(base, t_end=5.0, n_points=11).y)
    assert np.array_equal(tr_b.y, simulate(other, t_end=5.0, n_points=11).y)


# ------------------------------------------------------------- scan failure isolation

def test_scan_isolates_per_value_failures_offline():
    net = synthesis_network()
    values = [0.5, -1.0, 2.0]  # middle value is an illegal rate constant
    outcomes = scan_parameter_isolated(net, "synth", "k", values,
                                       t_end=5.0, n_points=11)
    for o in outcomes:
        log.info("value=%s error=%s", o.value,
                 None if o.error is None else (o.error.code, o.error.detail))
    assert [o.value for o in outcomes] == values
    assert outcomes[0].error is None and outcomes[0].trajectory is not None
    assert outcomes[1].error is not None
    assert outcomes[1].error.code == ErrorCode.INFEASIBLE_RATE
    assert "synth" in outcomes[1].error.detail
    assert outcomes[2].error is None and outcomes[2].trajectory is not None
    # every point (even failed) carries its own fingerprint
    assert len({o.fingerprint for o in outcomes}) == 3


def test_scan_endpoint_returns_per_point_errors():
    client = TestClient(create_app())
    net = {
        "species": {"A": 2.0, "B": 1.0, "C": 0.0},
        "reactions": [{"name": "synth", "reactants": {"A": 2, "B": 1},
                       "products": {"C": 1},
                       "rate": {"kind": "mass_action", "params": {"k": 0.5}}}],
        "compositions": {"A": {"C": 1, "H": 1}, "B": {"C": 1},
                         "C": {"C": 3, "H": 2}},
    }
    resp = client.post("/scan", json={
        "network": net, "reaction": "synth", "param": "k",
        "values": [0.5, -1.0, 2.0], "t_end": 5.0, "n_points": 11,
    })
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    log.info("scan results: %s",
             [(r["value"], "error" if "error" in r else "ok") for r in results])
    assert "trajectory" in results[0] and "error" not in results[0]
    assert results[1]["error"]["code"] == "INFEASIBLE_RATE"
    assert "synth" in results[1]["error"]["detail"]
    assert "trajectory" in results[2]
    # failed point does not poison the others' trajectories
    assert results[0]["trajectory"]["y"] != results[2]["trajectory"]["y"]


def test_scan_unknown_parameter_still_raises():
    net = synthesis_network()
    with pytest.raises(NetworkValidationError) as excinfo:
        scan_parameter_isolated(net, "synth", "nope", [1.0], t_end=1.0)
    log.info("unknown param -> %s", excinfo.value.code)
    assert excinfo.value.code == ErrorCode.UNKNOWN_PARAMETER


def test_concurrent_scans_on_composition_network_are_isolated():
    net = synthesis_network()
    ks = [0.25, 0.5, 1.0, 2.0, 4.0]

    def one_scan(k):
        outcome = scan_parameter_isolated(net, "synth", "k", [k],
                                          t_end=5.0, n_points=21)[0]
        fp = network_fingerprint(perturb_parameter(net, "synth", "k", k))
        return k, fp, outcome.trajectory.y

    with ThreadPoolExecutor(max_workers=12) as pool:
        rounds = list(pool.map(lambda _: list(pool.map(one_scan, ks)), range(8)))
    baseline = {k: y for k, _fp, y in rounds[0]}
    for round_i, round_results in enumerate(rounds):
        for k, fp, y in round_results:
            assert np.array_equal(y, baseline[k]), f"round {round_i} k={k}"
    distinct = {fp for rr in rounds for _k, fp, _y in rr}
    log.info("8 concurrent rounds x %d values -> %d distinct fingerprints",
             len(ks), len(distinct))
    assert len(distinct) == len(ks)
    assert net.reactions[0].rate.params["k"] == 0.5  # source untouched
