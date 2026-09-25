"""End-to-end HTTP tests: concurrent requests and error envelopes."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.service import create_app

log = logging.getLogger("test.api")

NET = {
    "species": {"A": 1.0, "B": 0.0},
    "reactions": [
        {
            "name": "r",
            "reactants": {"A": 1},
            "products": {"B": 1},
            "rate": {"kind": "mass_action", "params": {"k": 1.0}},
            "reverse": {"kind": "mass_action", "params": {"k": 0.5}},
        }
    ],
    "version": "demo-1",
}


def test_simulate_ok_and_envelope():
    client = TestClient(create_app())
    resp = client.post("/simulate",
                       json={"network": NET, "t_end": 10.0, "n_points": 11})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    log.info("fingerprint=%s final_state=%s",
             body["fingerprint"][:12], body["trajectory"]["y"][-1])
    assert "fingerprint" in body and "trajectory" in body
    assert len(body["trajectory"]["t"]) == 11


def test_concurrent_simulate_requests_are_consistent():
    client = TestClient(create_app())

    def one_call(i):
        payload = {"network": NET, "t_end": 10.0, "n_points": 101}
        r = client.post("/simulate", json=payload)
        return i, r.status_code, r.json()["trajectory"]["y"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        answers = list(pool.map(one_call, range(32)))
    ref = answers[0][2]
    for i, status, y in answers:
        assert status == 200
        assert y == ref, f"request {i} diverged"
    log.info("32 concurrent identical requests -> identical trajectories")


def test_scan_endpoint_uses_distinct_fingerprints():
    client = TestClient(create_app())
    resp = client.post("/scan", json={
        "network": NET, "reaction": "r", "param": "k",
        "values": [0.5, 1.0, 2.0], "t_end": 10.0, "n_points": 11,
    })
    assert resp.status_code == 200, resp.text
    fps = [r["fingerprint"] for r in resp.json()["results"]]
    finals = [r["trajectory"]["y"][-1][1] for r in resp.json()["results"]]
    log.info("scan fingerprints: %s", [f[:8] for f in fps])
    log.info("final [B] per k: %s", finals)
    assert len(set(fps)) == 3
    assert finals == sorted(finals)  # larger k -> more B


def test_error_envelope_is_machine_readable():
    client = TestClient(create_app())
    bad = json.loads(json.dumps(NET))
    bad["reactions"][0]["reactants"] = {"GHOST": 1}
    resp = client.post("/simulate", json={"network": bad, "t_end": 1.0})
    log.info("bad request -> %s %s", resp.status_code, resp.json())
    assert resp.status_code == 422
    assert resp.json()["code"] == "MISSING_SPECIES"
