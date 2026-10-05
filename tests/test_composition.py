"""Composition bookkeeping: multi-substrate/non-1:1 balance, multiple
independent conservation laws, invalid/contradictory composition, and
legacy (composition-free) compatibility.

Every case logs its input network and the exact evidence used for the
verdict (per-component totals, per-law drift, fingerprints); failure
messages name the reaction plus component or the conservation law.
"""

import logging

from app.models import Network, RateLaw, Reaction

log = logging.getLogger("test.composition")


def ma(k=1.0):
    return RateLaw("mass_action", {"k": k})


def _two_a_plus_b(comp_c=4):
    """2A + B -> C with x: A=1, B=2, balanced C=4."""
    return Network(
        species={"A": 2.0, "B": 1.0, "C": 0.0},
        reactions=(Reaction("r1", {"A": 2, "B": 1}, {"C": 1}, ma(1.0), ma(0.5)),),
        composition={"A": {"x": 1}, "B": {"x": 2}, "C": {"x": comp_c}},
        version="2ab-c",
    )


def test_two_substrates_nonunit_stoichiometry_balanced():
    net = _two_a_plus_b(comp_c=4)
    log.info("input: 2A+B->C composition=%s", net.composition)
    from app.validation import validate_network
    validate_network(net)  # must not raise
    log.info("verdict: balanced (reactant x = 2*1+1*2 = 4 = product C.x)")


def test_cleavage_one_to_many_balanced():
    net = Network(
        species={"A": 1.0, "B": 0.0, "C": 0.0},
        reactions=(Reaction("split", {"A": 1}, {"B": 1, "C": 1}, ma(1.0)),),
        composition={"A": {"x": 1, "y": 1}, "B": {"x": 1}, "C": {"y": 1}},
        version="split",
    )
    log.info("input: A->B+C composition=%s", net.composition)
    from app.validation import validate_network
    validate_network(net)
    log.info("verdict: x: 1=1 and y: 1=1, balanced")


def test_three_substrates_with_group_coefficients_balanced():
    # 2A + 3B + C -> P ; x counts: A=1,B=2,C=3 -> 2+6+3=11 = P
    net = Network(
        species={"A": 2.0, "B": 3.0, "C": 1.0, "P": 0.0},
        reactions=(Reaction("build", {"A": 2, "B": 3, "C": 1}, {"P": 1}, ma(1.0)),),
        composition={"A": {"x": 1}, "B": {"x": 2}, "C": {"x": 3}, "P": {"x": 11}},
    )
    log.info("input: 2A+3B+C->P, left x=2*1+3*2+1*3=11, P.x=11")
    from app.validation import validate_network
    validate_network(net)


def test_unbalanced_names_reaction_direction_component_and_entries():
    import pytest

    from app.errors import ErrorCode, NetworkValidationError
    net = _two_a_plus_b(comp_c=3)  # product side short by one x
    log.info("input: %s with C.x=3 (left x=4)", net.reactions[0].name)
    with pytest.raises(NetworkValidationError) as exc:
        from app.validation import validate_network
        validate_network(net)
    log.info("got code=%s detail=%s", exc.value.code, exc.value.detail)
    assert exc.value.code is ErrorCode.UNBALANCED_REACTION
    detail = exc.value.detail
    assert "r1" in detail and "forward" in detail and "'x'" in detail
    assert "4" in detail and "3" in detail  # exact side totals
    assert "C(3)" in detail  # offending composition entry pinpointed


def test_imbalance_message_names_reaction_and_component():
    """A reversible reaction missing component y on one side is rejected;
    the message names the reaction and the offending component and is
    never silently corrected."""
    import pytest

    from app.errors import ErrorCode, NetworkValidationError
    net = Network(
        species={"A": 1.0, "B": 0.0},
        reactions=(Reaction("r", {"A": 1}, {"B": 1}, ma(1.0), ma(0.1)),),
        composition={"A": {"x": 1, "y": 1}, "B": {"x": 1}},  # y vanishes
    )
    log.info("input: A<->B with A={x,y}, B={x}; y cannot be destroyed")
    with pytest.raises(NetworkValidationError) as exc:
        from app.validation import validate_network
        validate_network(net)
    log.info("code=%s detail=%s", exc.value.code, exc.value.detail)
    assert exc.value.code is ErrorCode.UNBALANCED_REACTION
    assert "'y'" in exc.value.detail and "r" in exc.value.detail
    assert "reactant side has 1" in exc.value.detail
    assert "product side has 0" in exc.value.detail


def test_balanced_multisubstrate_network_simulates():
    from app.simulation import simulate
    net = _two_a_plus_b()
    tr = simulate(net, t_end=5.0, n_points=11)
    log.info("simulated 2A+B<->C final=%s", dict(zip(tr.species, tr.y[-1])))
    assert all(d.within_tolerance for d in tr.conservation_laws)


def _two_component_network():
    """A -> B + C: x = A+B, y = A+C; two independent conservation laws."""
    return Network(
        species={"A": 1.0, "B": 0.0, "C": 0.0},
        reactions=(
            Reaction("split", {"A": 1}, {"B": 1, "C": 1}, ma(2.0), ma(1.0)),
        ),
        composition={"A": {"x": 1, "y": 1}, "B": {"x": 1}, "C": {"y": 1}},
        version="two-cons",
    )


def test_each_independent_conservation_law_is_reported():
    from app.composition import conservation_laws
    net = _two_component_network()
    laws = {l.name: l.weights for l in conservation_laws(net, ("A", "B", "C"))}
    log.info("reported laws: %s", laws)
    assert laws["composition:x"] == {"A": 1, "B": 1}
    assert laws["composition:y"] == {"A": 1, "C": 1}


def test_simulation_reports_drift_of_every_law():
    from app.simulation import simulate
    net = _two_component_network()
    tr = simulate(net, t_end=10.0, n_points=21)
    by_name = {d.name: d for d in tr.conservation_laws}
    log.info("x: initial=%.6f drift=%.3e | y: initial=%.6f drift=%.3e",
             by_name["composition:x"].initial_value,
             by_name["composition:x"].max_deviation,
             by_name["composition:y"].initial_value,
             by_name["composition:y"].max_deviation)
    assert len(tr.conservation_laws) == 2
    assert all(d.within_tolerance for d in tr.conservation_laws)
    assert by_name["composition:x"].initial_value == 1.0
    assert by_name["composition:y"].initial_value == 1.0


def test_violation_names_which_conservation_law_exceeded(monkeypatch):
    import numpy as np
    import pytest

    from app import simulation as sim_mod
    from app.errors import SimulationError
    net = _two_component_network()
    # x stays constant (A+B=1) while y drifts (A+C = 1 -> 1.01)
    t = np.linspace(0.0, 1.0, 2)
    y = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.51]])
    monkeypatch.setattr(sim_mod, "_integrate", lambda n, te: (t, y))
    log.info("injected trajectory=%s (x constant, y +%.3f)",
             y.tolist(), 0.01)
    with pytest.raises(SimulationError) as exc:
        sim_mod.simulate(net, t_end=1.0, n_points=2, conservation_tol=1e-6)
    log.info("code=%s detail=%s", exc.value.code, exc.value.detail)
    assert "composition:y" in exc.value.detail
    assert "composition:x" not in exc.value.detail


def test_conservation_verdict_invariant_to_grid_segmentation_and_concurrency():
    """Refining output steps, splitting the interval, and running many
    concurrent simulations must report identical laws and drifts."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from app.simulation import simulate
    net = _two_component_network()
    full = simulate(net, t_end=10.0, n_points=21)
    fine = simulate(net, t_end=10.0, n_points=201)
    seg1 = simulate(net, t_end=5.0, n_points=11)
    seg2 = simulate(net, t_end=10.0, t_start=5.0, n_points=11)
    log.info("fine-vs-coarse max y diff: %.3e",
             np.max(np.abs(fine.y[::10] - full.y)))
    assert np.allclose(fine.y[::10], full.y, atol=1e-10)
    assert np.allclose(seg1.y, full.y[:11], atol=1e-10)
    assert np.allclose(seg2.y, full.y[10:], atol=1e-10)

    def drifts_of(_):
        return [(d.name, round(d.max_deviation, 18))
                for d in simulate(net, t_end=10.0, n_points=21).conservation_laws]

    with ThreadPoolExecutor(max_workers=8) as pool:
        rounds = list(pool.map(drifts_of, range(16)))
    assert all(r == rounds[0] for r in rounds)
    log.info("16 concurrent runs: identical law drifts %s", rounds[0])


def _composed_species(**entries):
    return Network(species={"A": 1.0, "B": 0.0}, reactions=(), composition=entries)


INVALID_COMPOSITION_CASES = {
    "composition_for_undeclared_species": (
        lambda: Network(species={"A": 1.0}, reactions=(),
                        composition={"A": {"x": 1}, "GHOST": {"x": 1}}),
        "GHOST",
    ),
    "missing_declaration_for_one_species": (
        lambda: _composed_species(A={"x": 1}),
        "'B'",
    ),
    "zero_component_count": (
        lambda: _composed_species(A={"x": 0}, B={"x": 1}),
        "illegal count",
    ),
    "negative_component_count": (
        lambda: _composed_species(A={"x": -2}, B={"x": 1}),
        "illegal count",
    ),
    "non_finite_component_count": (
        lambda: _composed_species(A={"x": float("nan")}, B={"x": 1}),
        "illegal count",
    ),
    "empty_component_map": (
        lambda: _composed_species(A={}, B={"x": 1}),
        "empty or non-object",
    ),
}


def test_invalid_composition_raises_with_locatable_detail():
    import pytest

    from app.errors import ErrorCode, NetworkValidationError
    from app.validation import validate_network
    for label, (factory, needle) in INVALID_COMPOSITION_CASES.items():
        net = factory()
        log.info("case=%s input composition=%s", label, net.composition)
        with pytest.raises(NetworkValidationError) as exc:
            validate_network(net)
        log.info("  -> %s: %s", exc.value.code, exc.value.detail)
        assert exc.value.code is ErrorCode.INVALID_COMPOSITION, label
        assert needle in exc.value.detail, label


def test_legacy_network_without_composition_is_unchanged():
    """No composition field: validation verdict, fingerprint and results
    must be exactly as before the feature existed."""
    from app.simulation import network_fingerprint, simulate
    legacy = Network(
        species={"A": 1.0, "B": 0.0},
        reactions=(Reaction("r", {"A": 1}, {"B": 1}, ma(2.0), ma(1.0)),),
        version="ab-v1",
    )  # composition defaults to {} -- indistinguishable from old clients
    from app.validation import validate_network
    validate_network(legacy)
    fp = network_fingerprint(legacy)
    log.info("legacy fingerprint=%s", fp)
    # exact pre-feature fingerprint: absence of composition must not
    # perturb the canonical payload by a single byte
    assert fp == "a6a9a85b5dc31507b11e7b824bf5f9b2debfc7f88a674d8c0873f8ed3a67ee02"
    tr = simulate(legacy, t_end=20.0, n_points=21)
    a_eq, b_eq = tr.y[-1]
    log.info("legacy equilibrium [A]=%.6f [B]=%.6f", a_eq, b_eq)
    assert abs(a_eq - 1 / 3) < 1e-6 and abs(b_eq - 2 / 3) < 1e-6
    assert tr.conserved_total == 1.0
    assert tr.max_conservation_deviation < 1e-9


def test_same_species_names_different_composition_distinct_fingerprint():
    from app.simulation import network_fingerprint
    from app.validation import validate_network
    species = {"A": 1.0, "B": 0.0}
    rxn = (Reaction("r", {"A": 1}, {"B": 1}, ma(1.0)),)
    net_x = Network(species=species, reactions=rxn,
                    composition={"A": {"x": 1}, "B": {"x": 1}})
    net_y = Network(species=species, reactions=rxn,
                    composition={"A": {"y": 1}, "B": {"y": 1}})
    validate_network(net_x)
    validate_network(net_y)
    fp_x, fp_y = network_fingerprint(net_x), network_fingerprint(net_y)
    log.info("same names, x-composition fp=%s, y-composition fp=%s",
             fp_x[:12], fp_y[:12])
    assert fp_x != fp_y


def test_legacy_nonunit_stoichiometry_is_validated_and_simulates():
    """The old total-species-sum check rejected balanced 2A+B->C as a
    false positive. Without composition the structural conservation law
    2A+B+C=const is now used, so the network passes and simulates."""
    from app.simulation import simulate
    net = Network(
        species={"A": 2.0, "B": 1.0, "C": 0.0},
        reactions=(Reaction("r", {"A": 2, "B": 1}, {"C": 1}, ma(1.0)),),
        version="legacy-nonunit",
    )
    from app.validation import validate_network
    validate_network(net)
    tr = simulate(net, t_end=5.0, n_points=11)
    log.info("legacy non-1:1 laws=%s final=%s",
             [d.weights for d in tr.conservation_laws], tr.y[-1])
    assert all(d.within_tolerance for d in tr.conservation_laws)


def test_scan_single_point_failure_is_isolated_offline():
    from app.scanning import safe_scan_parameter
    net = _two_a_plus_b()
    points = safe_scan_parameter(
        net, "r1", "k", [0.5, -1.0, 2.0], t_end=5.0, n_points=6
    )
    log.info("scan values=[0.5,-1,2] statuses=%s",
             [(p.value, p.ok, p.error_code) for p in points])
    assert [p.ok for p in points] == [True, False, True]
    assert points[1].error_code == "INFEASIBLE_RATE"
    assert "-1.0" in points[1].error_detail and "r1" in points[1].error_detail
    assert points[0].trajectory is not None and points[2].trajectory is not None
    assert points[0].fingerprint != points[2].fingerprint
    # source network untouched
    assert net.reactions[0].rate.params["k"] == 1.0
    log.info("source k still %s; neighboring points unaffected",
             net.reactions[0].rate.params["k"])


def test_scan_endpoint_isolates_failed_point():
    from fastapi.testclient import TestClient

    from app.service import create_app
    client = TestClient(create_app())
    payload = {
        "species": {"A": 2.0, "B": 1.0, "C": 0.0},
        "composition": {"A": {"x": 1}, "B": {"x": 2}, "C": {"x": 4}},
        "reactions": [{
            "name": "r1", "reactants": {"A": 2, "B": 1}, "products": {"C": 1},
            "rate": {"kind": "mass_action", "params": {"k": 1.0}},
            "reverse": {"kind": "mass_action", "params": {"k": 0.5}},
        }],
    }
    resp = client.post("/scan", json={
        "network": payload, "reaction": "r1", "param": "k",
        "values": [0.5, -1.0, 2.0], "t_end": 5.0, "n_points": 6,
    })
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    log.info("http scan results: %s",
             [(r["value"], r["ok"], r.get("code")) for r in results])
    assert [r["ok"] for r in results] == [True, False, True]
    assert results[1]["code"] == "INFEASIBLE_RATE"
    assert "r1" in results[1]["detail"]
    assert results[0]["fingerprint"] != results[2]["fingerprint"]


def test_concurrent_scans_over_new_networks_do_not_pollute():
    from concurrent.futures import ThreadPoolExecutor

    from app.scanning import safe_scan_parameter
    net = _two_a_plus_b()
    values = [0.25, -1.0, 0.5, 2.0, 4.0]

    def one_round(_):
        return [
            (p.value, p.ok, p.error_code, p.fingerprint)
            for p in safe_scan_parameter(net, "r1", "k", values,
                                         t_end=5.0, n_points=6)
        ]

    with ThreadPoolExecutor(max_workers=10) as pool:
        rounds = list(pool.map(one_round, range(10)))
    for r in rounds:
        assert [x[:3] for x in r] == [x[:3] for x in rounds[0]]
    log.info("10 concurrent scans x 5 points: same verdicts %s",
             [(v, ok, code) for v, ok, code, _ in rounds[0]])
    # identical good values share one fingerprint within and across rounds;
    # failed values carry no fingerprint and cannot poison a good point
    good_fps = {v: fp for v, ok, _, fp in rounds[0] if ok}
    for r in rounds:
        for v, ok, _, fp in r:
            if ok:
                assert fp == good_fps[v]
    assert net.reactions[0].rate.params["k"] == 1.0


def test_scan_reports_conservation_failure_at_a_single_value(monkeypatch):
    """If one parameter value yields a drifting trajectory, only that
    point is marked CONSERVATION_VIOLATION naming the law."""
    from app import simulation as sim_mod
    from app.scanning import safe_scan_parameter
    net = _two_component_network()
    real_integrate = sim_mod._integrate

    def fake(network, t_end):
        k = network.reactions[0].rate.params["k"]
        t, y = real_integrate(network, t_end)
        if k == 3.0:
            y = y.copy()
            # ramp the break so the t=0 value is untouched but the
            # trajectory later violates conservation y = A + C
            y[:, 2] += 0.01 * (t / t_end)
        return t, y

    monkeypatch.setattr(sim_mod, "_integrate", fake)
    points = safe_scan_parameter(net, "split", "k", [1.0, 3.0],
                                 t_end=5.0, n_points=6)
    log.info("points: %s", [(p.value, p.ok, p.error_code) for p in points])
    assert points[0].ok and not points[1].ok
    assert points[1].error_code == "CONSERVATION_VIOLATION"
    assert "composition:y" in points[1].error_detail


def test_open_network_source_sink_exempt_but_internal_reactions_must_balance():
    import pytest

    from app.errors import ErrorCode, NetworkValidationError
    from app.validation import validate_network
    # feed: 0 -> A(boundary), legitimate exchange, accepted
    open_net = Network(
        species={"A": 1.0},
        reactions=(Reaction("feed", {}, {"A": 1}, ma(0.5)),),
        boundary_species=frozenset({"A"}),
        composition={"A": {"x": 1}},
    )
    validate_network(open_net)
    log.info("open source 0->A(boundary) accepted as environment exchange")
    # an internal A -> B + C reaction that creates an extra x is still
    # rejected even though the network is open
    bad_internal = Network(
        species={"A": 1.0, "B": 0.0, "C": 0.0},
        reactions=(Reaction("r", {"A": 1}, {"B": 1, "C": 1}, ma(1.0)),),
        boundary_species=frozenset({"C"}),
        composition={"A": {"x": 1}, "B": {"x": 1}, "C": {"x": 1}},
    )
    with pytest.raises(NetworkValidationError) as exc:
        validate_network(bad_internal)
    log.info("internal imbalance in open net -> %s", exc.value.detail)
    assert exc.value.code is ErrorCode.UNBALANCED_REACTION


def test_scan_with_unknown_parameter_fails_every_point_locatably():
    """A bad scan target (nonexistent parameter) must not 500 or abort:
    every point comes back failed with UNKNOWN_PARAMETER naming it."""
    from fastapi.testclient import TestClient

    from app.service import create_app
    client = TestClient(create_app())
    payload = {
        "species": {"A": 2.0, "B": 1.0, "C": 0.0},
        "composition": {"A": {"x": 1}, "B": {"x": 2}, "C": {"x": 4}},
        "reactions": [{
            "name": "r1", "reactants": {"A": 2, "B": 1}, "products": {"C": 1},
            "rate": {"kind": "mass_action", "params": {"k": 1.0}},
        }],
    }
    resp = client.post("/scan", json={
        "network": payload, "reaction": "r1", "param": "nope",
        "values": [1.0, 2.0], "t_end": 5.0, "n_points": 6,
    })
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    log.info("unknown-param scan: %s",
             [(r["value"], r["ok"], r.get("code")) for r in results])
    assert all(r["ok"] is False for r in results)
    assert all(r["code"] == "UNKNOWN_PARAMETER" for r in results)
    assert all("'nope'" in r["detail"] for r in results)
