"""Illegal networks: each failure cause has a distinct code and is never fixed."""

from __future__ import annotations

import logging
import uuid

import pytest

from app.errors import NetworkValidationError
from app.parser import parse_network
from app.validation import validate_network
from tests.conftest import CLOSED_A_B, ENZYME_NET

log = logging.getLogger("bionet.tests")

CASE = uuid.uuid4().hex[:8]  # marker making this run's key inputs findable


def _reject(payload: dict, expected_code: str) -> NetworkValidationError:
    log.info("[%s] expecting code=%s payload=%s", CASE, expected_code, payload)
    try:
        network = parse_network(payload)
        validate_network(network)
    except NetworkValidationError as exc:
        log.info("[%s] rejected code=%s message=%s details=%s",
                 CASE, exc.code, exc.message, exc.details)
        assert exc.code == expected_code
        assert exc.details, "errors must carry locatable details"
        return exc
    raise AssertionError(f"expected {expected_code} but network was accepted")


def test_missing_species():
    payload = {
        "model_id": "x",
        "species": [{"id": "A", "initial_concentration": 1.0}],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"Z": 1},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    exc = _reject(payload, "missing_species")
    assert exc.details["species"] == "Z"


def test_missing_enzyme_species():
    payload = {
        "model_id": "x",
        "species": [
            {"id": "S", "initial_concentration": 1.0, "composition": {"M": 1}},
            {"id": "P", "initial_concentration": 0.0, "composition": {"M": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"S": 1}, "products": {"P": 1},
             "forward": {"kind": "enzymatic", "k": 1.0,
                         "enzyme": "E", "km": 1.0}}
        ],
    }
    _reject(payload, "missing_species")


def test_negative_stoichiometry():
    payload = {
        "model_id": "x",
        "species": [
            {"id": "A", "initial_concentration": 1.0, "composition": {"X": 1}},
            {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": -1}, "products": {"B": 1},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    _reject(payload, "invalid_stoichiometry")


def test_zero_net_stoichiometry():
    payload = {
        "model_id": "x",
        "species": [{"id": "A", "initial_concentration": 1.0}],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"A": 1},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    _reject(payload, "invalid_stoichiometry")


def test_missing_rate_constant():
    payload = {
        "model_id": "x",
        "species": [
            {"id": "A", "initial_concentration": 1.0, "composition": {"X": 1}},
            {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
             "forward": {"kind": "mass_action"}}
        ],
    }
    _reject(payload, "invalid_rate")


def test_reversible_without_reverse_law():
    payload = {
        "model_id": "x",
        "species": [
            {"id": "A", "initial_concentration": 1.0, "composition": {"X": 1}},
            {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
             "reversible": True,
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    _reject(payload, "invalid_rate")


def test_enzyme_absent_initially():
    payload = {
        "model_id": "x",
        "species": [
            {"id": "S", "initial_concentration": 1.0, "composition": {"M": 1}},
            {"id": "P", "initial_concentration": 0.0, "composition": {"M": 1}},
            {"id": "E", "initial_concentration": 0.0},
        ],
        "reactions": [
            {"id": "r", "reactants": {"S": 1}, "products": {"P": 1},
             "forward": {"kind": "enzymatic", "k": 1.0,
                         "enzyme": "E", "km": 1.0}}
        ],
    }
    _reject(payload, "invalid_rate")


def test_unclosed_network():
    payload = {
        "model_id": "x",
        "species": [
            {"id": "A", "initial_concentration": 1.0, "composition": {"X": 1}},
            {"id": "B", "initial_concentration": 0.0, "composition": {"Y": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    exc = _reject(payload, "unclosed_network")
    assert "imbalance" in exc.details


def test_source_reaction_unclosed():
    payload = {
        "model_id": "x",
        "species": [
            {"id": "A", "initial_concentration": 0.0, "composition": {"X": 1}},
        ],
        "reactions": [
            {"id": "r", "reactants": {}, "products": {"A": 1},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    _reject(payload, "unclosed_network")


def test_negative_initial_concentration():
    payload = {
        "model_id": "x",
        "species": [{"id": "A", "initial_concentration": -1.0}],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    _reject(payload, "invalid_initial_state")


def test_validation_order_is_deterministic():
    """First fault in check order wins: missing species before closure."""
    payload = {
        "model_id": "x",
        "species": [{"id": "A", "initial_concentration": 1.0}],
        "reactions": [
            {"id": "r", "reactants": {"A": 1}, "products": {"Z": 2},
             "forward": {"kind": "mass_action", "k": 1.0}}
        ],
    }
    _reject(payload, "missing_species")


def test_valid_networks_accepted(service):
    for payload in (CLOSED_A_B, ENZYME_NET):
        validate_network(parse_network(payload))
