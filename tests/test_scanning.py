"""Parameter perturbation, reproducibility and concurrent-scan isolation."""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.models import Network, RateLaw, Reaction
from app.scanning import perturb_parameter, scan_parameter
from app.simulation import network_fingerprint, simulate

log = logging.getLogger("test.scan")


def irreversible_ab(k=1.0, version="ab-v1") -> Network:
    return Network(
        species={"A": 1.0, "B": 0.0},
        reactions=(
            Reaction(
                "r", {"A": 1}, {"B": 1}, RateLaw("mass_action", {"k": k})
            ),
        ),
        version=version,
    )


def test_perturbation_does_not_mutate_source_network():
    net = irreversible_ab(k=1.0)
    before_fingerprint = network_fingerprint(net)
    changed = perturb_parameter(net, "r", "k", 5.0)
    log.info("original k=%s; perturbed k=%s",
             net.reactions[0].rate.params["k"],
             changed.reactions[0].rate.params["k"])
    assert net.reactions[0].rate.params["k"] == 1.0
    assert changed.reactions[0].rate.params["k"] == 5.0
    assert network_fingerprint(net) == before_fingerprint
    assert network_fingerprint(changed) != before_fingerprint
    log.info("fingerprints differ: %s vs %s",
             before_fingerprint[:12], network_fingerprint(changed)[:12])


def test_different_versions_never_share_results():
    v1 = irreversible_ab(k=1.0, version="model-v1")
    v2 = irreversible_ab(k=1.0, version="model-v2")
    log.info("same kinetics, versions=%s/%s -> fingerprints must differ",
             v1.version, v2.version)
    assert network_fingerprint(v1) != network_fingerprint(v2)
    # simulations are still consistent within a version
    assert np.allclose(simulate(v1, 5.0, 6).y, simulate(v1, 5.0, 6).y)


def test_identical_requests_are_reproducible():
    net = irreversible_ab()
    first = simulate(net, t_end=5.0, n_points=21)
    second = simulate(net, t_end=5.0, n_points=21)
    log.info("same request twice; exact array equality: %s",
             np.array_equal(first.y, second.y))
    assert np.array_equal(first.y, second.y)


def test_scan_results_follow_parameter_monotonically():
    net = irreversible_ab()
    values = [0.25, 0.5, 1.0, 2.0, 4.0]
    results = scan_parameter(net, "r", "k", values, t_end=5.0, n_points=21)
    finals_b = [tr.y[-1, 1] for tr in results]
    log.info("scanned k=%s -> final [B]=%s", values,
             [round(x, 5) for x in finals_b])
    assert all(a < b for a, b in zip(finals_b, finals_b[1:]))
    # source network untouched after the scan
    assert net.reactions[0].rate.params["k"] == 1.0


def test_concurrent_scans_are_isolated_and_reproducible():
    net = irreversible_ab()
    ks = [0.25, 0.5, 1.0, 2.0, 4.0]

    def one_scan(k):
        tr = scan_parameter(net, "r", "k", [k], t_end=5.0, n_points=21)[0]
        return k, network_fingerprint(perturb_parameter(net, "r", "k", k)), tr.y

    with ThreadPoolExecutor(max_workers=12) as pool:
        rounds = list(pool.map(lambda _: list(pool.map(one_scan, ks)), range(8)))
    # 8 rounds x 5 values all interleaved concurrently
    baseline = {k: y for k, _fp, y in rounds[0]}
    for round_i, round_results in enumerate(rounds):
        for k, fp, y in round_results:
            assert np.array_equal(y, baseline[k]), f"round {round_i} k={k}"
    log.info("8 concurrent rounds x 5 k-values: every round matches baseline")
    log.info("source k after all scans: %s (must remain 1.0)",
             net.reactions[0].rate.params["k"])
    assert net.reactions[0].rate.params["k"] == 1.0
    # fingerprints are per-value, so cached results cannot cross-contaminate
    fps = {round_i: {} for round_i in range(8)}
    for round_i, round_results in enumerate(rounds):
        for k, fp, _y in round_results:
            fps[round_i][k] = fp
    distinct_fps = {fp for r in fps.values() for fp in r.values()}
    assert len(distinct_fps) == len(ks)
    log.info("%d distinct k values -> %d distinct fingerprints (no reuse)",
             len(ks), len(distinct_fps))
