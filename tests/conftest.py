"""Shared fixtures and logging for the biochemical network simulator tests."""

from __future__ import annotations

import logging

import pytest

from app.service import SimulationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bionet.tests")

CLOSED_A_B = {
    "model_id": "ab",
    "version": 1,
    "species": [
        {"id": "A", "initial_concentration": 2.0, "composition": {"X": 1}},
        {"id": "B", "initial_concentration": 0.0, "composition": {"X": 1}},
    ],
    "reactions": [
        {"id": "r", "reactants": {"A": 1}, "products": {"B": 1},
         "reversible": True,
         "forward": {"kind": "mass_action", "k": 1.0},
         "reverse": {"kind": "mass_action", "k": 3.0}}
    ],
}

ENZYME_NET = {
    "model_id": "enz",
    "version": 1,
    "species": [
        {"id": "S", "initial_concentration": 5.0, "composition": {"M": 1}},
        {"id": "P", "initial_concentration": 0.0, "composition": {"M": 1}},
        {"id": "E", "initial_concentration": 0.1},
    ],
    "reactions": [
        {"id": "cat", "reactants": {"S": 1}, "products": {"P": 1},
         "forward": {"kind": "enzymatic", "k": 2.0, "enzyme": "E", "km": 1.0}}
    ],
}


@pytest.fixture
def service() -> SimulationService:
    svc = SimulationService()
    svc.register_network(CLOSED_A_B)
    svc.register_network(ENZYME_NET)
    return svc
