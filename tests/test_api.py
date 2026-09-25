"""HTTP entry point: error codes, batch vs single validation parity, scan API."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

import main
from tests.conftest import CLOSED_A_B, ENZYME_NET

log = logging.getLogger("bionet.tests")


@pytest.fixture
def client():
    svc = type(main.service)()
    main.service = svc
    return TestClient(main.app)


def test_register_and_run(client):
    resp = client.post("/networks", json={"network": CLOSED_A_B})
    assert resp.status_code == 200
    resp = client.post("/simulate", json={
        "model_id": "ab", "config": {"t_end": 30, "output_step": 1}})
    body = resp.json()
    log.info("api run steady=%s drift=%.3e",
             body["steady_state"], body["max_conservation_error"])
    assert resp.status_code == 200
    assert body["steady_state"] is not None


def test_error_codes_are_distinct_and_locatable(client):
    cases = [
        ({"model_id": "x",
          "species": [{"id": "A", "initial_concentration": 1}],
          "reactions": [{"id": "r", "reactants": {"A": 1},
                         "products": {"Z": 1},
                         "forward": {"kind": "mass_action", "k": 1}}]},
         "missing_species"),
        ({"model_id": "x",
          "species": [{"id": "A", "initial_concentration": 1}],
          "reactions": [{"id": "r", "reactants": {"A": -1}, "products": {},
                         "forward": {"kind": "mass_action", "k": 1}}]},
         "invalid_stoichiometry"),
        ({"model_id": "x",
          "species": [{"id": "A", "initial_concentration": 1}],
          "reactions": [{"id": "r", "reactants": {"A": 1}, "products": {},
                         "forward": {"kind": "mass_action"}}]},
         "invalid_rate"),
    ]
    for payload, code in cases:
        r = client.post("/networks", json={"network": payload})
        log.info("api reject input=%s -> %s %s",
                 payload["reactions"], r.status_code, r.json()["error"]["code"])
        assert r.status_code == 400
        assert r.json()["error"]["code"] == code


def test_unclosed_network_rejected_at_api(client):
    payload = {
        "model_id": "x",
        "species": [{"id": "A", "initial_concentration": 1, "composition": {"X": 1}},
                    {"id": "B", "initial_concentration": 0, "composition": {"Y": 1}}],
        "reactions": [{"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
                       "forward": {"kind": "mass_action", "k": 1}}],
    }
    r = client.post("/networks", json={"network": payload})
    assert r.json()["error"]["code"] == "unclosed_network"


def test_single_and_scan_validation_conclusions_agree(client):
    """The same invalid network is rejected identically by both entry paths."""
    bad = {
        "model_id": "bad",
        "species": [{"id": "A", "initial_concentration": 1, "composition": {"X": 1}},
                    {"id": "B", "initial_concentration": 0, "composition": {"Y": 1}}],
        "reactions": [{"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
                       "forward": {"kind": "mass_action", "k": 1}}],
    }
    r1 = client.post("/networks", json={"network": bad})
    log.info("register invalid -> %s", r1.json()["error"]["code"])
    # scan path cannot even start on an unregistered model
    assert r1.status_code == 400

    # valid network: a scan value producing an invalid perturbed network must
    # report the same code a direct single run with that override reports
    client.post("/networks", json={"network": CLOSED_A_B})
    single = client.post("/simulate", json={
        "model_id": "ab", "config": {"t_end": 10, "output_step": 1},
        "overrides": {"species.A.initial_concentration": -3.0}})
    scan = client.post("/scan", json={
        "model_id": "ab", "config": {"t_end": 10, "output_step": 1},
        "parameter_path": "species.A.initial_concentration", "values": [-3.0]})
    log.info("parity single=%s scan_point=%s",
             single.json()["error"]["code"],
             scan.json()["points"][0]["error_code"])
    assert single.json()["error"]["code"] == "invalid_initial_state"
    assert scan.json()["points"][0]["error_code"] == "invalid_initial_state"


def test_scan_endpoint_returns_all_points(client):
    client.post("/networks", json={"network": ENZYME_NET})
    r = client.post("/scan", json={
        "model_id": "enz", "config": {"t_end": 20, "output_step": 1},
        "parameter_path": "reactions.cat.forward.km",
        "values": [0.25, 1.0, 4.0]})
    points = r.json()["points"]
    assert len(points) == 3
    assert all(p["ok"] for p in points)
    log.info("enzyme scan finals: %s",
             [p["result"]["steady_state"] for p in points])


def test_unknown_model_404(client):
    r = client.post("/simulate", json={
        "model_id": "nope", "config": {"t_end": 1, "output_step": 1}})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_model"
