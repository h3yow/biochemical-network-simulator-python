"""State isolation: overrides, scans, concurrency and model versions."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.parser import parse_sim_config

log = logging.getLogger("bionet.tests")

CFG = parse_sim_config({"t_end": 30.0, "output_step": 1.0})


def test_single_override_does_not_mutate_registered_model(service):
    baseline1 = service.run("ab", 1, CFG)
    perturbed = service.run("ab", 1, CFG,
                            {"reactions.r.forward.k": 9.0})
    baseline2 = service.run("ab", 1, CFG)
    log.info("override isolation: baseline final=%s perturbed final=%s",
             baseline1.values[-1], perturbed.values[-1])
    assert baseline1.values[-1] == baseline2.values[-1]
    assert abs(perturbed.values[-1][0] - baseline1.values[-1][0]) > 1e-6


def test_scan_points_are_independent(service):
    values = [0.25, 0.5, 1.0, 2.0, 4.0]
    points = service.scan("ab", 1, CFG, "reactions.r.forward.k", values)
    finals_a = [p.result.values[-1][0] for p in points]
    log.info("scan finals A for k=%s: %s", values, finals_a)
    assert all(p.error_code is None for p in points)
    # higher forward rate -> less A at equilibrium, strictly decreasing
    assert all(finals_a[i] > finals_a[i + 1] for i in range(len(finals_a) - 1))
    # baseline untouched afterwards
    assert abs(service.run("ab", 1, CFG).values[-1][0] - 1.5) < 1e-12


def test_scan_reports_bad_value_locally(service):
    points = service.scan("ab", 1, CFG,
                          "species.A.initial_concentration", [1.0, -2.0, 3.0])
    status = [(p.value, p.error_code) for p in points]
    log.info("scan with bad value: %s", status)
    assert points[0].error_code is None
    assert points[1].error_code == "invalid_initial_state"
    assert points[2].error_code is None


def test_concurrent_requests_do_not_pollute(service):
    barrier = threading.Barrier(8)

    def worker(k: float):
        barrier.wait()
        r = service.run("ab", 1, CFG, {"reactions.r.forward.k": k})
        # equilibrium of A<->B with kr=3: A = total * 3/(k+3)
        expected = 2.0 * 3.0 / (k + 3.0)
        return k, r.values[-1][0], expected

    ks = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, k) for k in ks]
        results = [f.result() for f in as_completed(futures)]
    for k, got, expected in results:
        log.info("concurrent k=%-6g got_A=%.8f expected_A=%.8f match=%s",
                 k, got, expected, abs(got - expected) < 1e-6)
        assert abs(got - expected) < 1e-6


def test_concurrent_scan_runs(service):
    """Many concurrent scans over different parameters must stay isolated."""
    payloads = [
        ("reactions.r.forward.k", [0.2, 0.8, 3.0, 12.0]),
        ("reactions.r.reverse.k", [0.5, 1.5, 6.0, 20.0]),
        ("species.A.initial_concentration", [0.5, 1.0, 4.0, 8.0]),
    ]

    def job(path, values):
        return service.scan("ab", 1, CFG, path, values)

    with ThreadPoolExecutor(max_workers=3) as pool:
        all_runs = list(pool.map(lambda args: job(*args), payloads))
    for (path, values), points in zip(payloads, all_runs):
        assert [p.value for p in points] == values
        assert all(p.error_code is None for p in points), (
            path, [p.error_code for p in points])
        log.info("concurrent scan path=%s finals=%s", path,
                 [round(p.result.values[-1][0], 6) for p in points])
    # cross-check: first scan identical to a serial rerun
    serial = service.scan("ab", 1, CFG, *payloads[0])
    for a, b in zip(all_runs[0], serial):
        assert a.result.values == b.result.values


def test_different_versions_are_not_reused(service):
    v2 = {
        "model_id": "ab", "version": 2,
        "species": [
            {"id": "A", "initial_concentration": 2.0, "composition": {"X": 1}},
            {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
             "reversible": True,
             "forward": {"kind": "mass_action", "k": 7.0},
             "reverse": {"kind": "mass_action", "k": 1.0}},
        ],
    }
    service.register_network(v2)
    r1 = service.run("ab", 1, CFG)
    r2 = service.run("ab", 2, CFG)
    latest = service.run("ab", None, CFG)
    log.info("version v1 final=%s v2 final=%s latest==v2:%s",
             r1.values[-1], r2.values[-1],
             latest.values[-1] == r2.values[-1])
    assert abs(r1.values[-1][0] - 1.5) < 1e-6
    assert abs(r2.values[-1][0] - 0.25) < 1e-6
    assert latest.values[-1] == r2.values[-1]


def test_same_version_different_content_rejected(service):
    from app.errors import SimError
    clone = {
        "model_id": "ab", "version": 1,
        "species": [
            {"id": "A", "initial_concentration": 9.0, "composition": {"X": 1}},
            {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
             "forward": {"kind": "mass_action", "k": 1.0}},
        ],
    }
    try:
        service.register_network(clone)
    except SimError as exc:
        assert exc.code == "sim_error"
        log.info("conflicting registration rejected: %s", exc.message)
    else:
        raise AssertionError("conflicting version content must be rejected")
    # identical content re-registration is idempotent
    from tests.conftest import CLOSED_A_B
    service.register_network(CLOSED_A_B)
