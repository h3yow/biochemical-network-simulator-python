"""Illegal-network validation: each cause maps to a distinct error code,
and the HTTP entry point must agree with the offline validator."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.errors import ErrorCode, NetworkValidationError
from app.models import Network, RateLaw, Reaction
from app.service import create_app
from app.validation import validate_network

log = logging.getLogger("test.validation")

MA = lambda k: RateLaw("mass_action", {"k": k})  # noqa: E731


def _net(reactions, species={"A": 1.0, "B": 0.0}, boundary=()):
    return Network(species=species, reactions=tuple(reactions),
                   boundary_species=frozenset(boundary))


CASES = {
    "missing_species": (
        _net([Reaction("r", {"A": 1, "X": 1}, {"B": 1}, MA(1.0))]),
        ErrorCode.MISSING_SPECIES,
    ),
    "missing_enzyme": (
        _net([Reaction("r", {"A": 1}, {"B": 1},
                       RateLaw("michaelis_menten",
                               {"kcat": 1.0, "Km": 0.5, "enzyme": "E"}))]),
        ErrorCode.MISSING_SPECIES,
    ),
    "negative_stoichiometry": (
        _net([Reaction("r", {"A": -1.0}, {"B": 1}, MA(1.0))]),
        ErrorCode.ILLEGAL_STOICHIOMETRY,
    ),
    "zero_stoichiometry": (
        _net([Reaction("r", {"A": 0.0}, {"B": 1}, MA(1.0))]),
        ErrorCode.ILLEGAL_STOICHIOMETRY,
    ),
    "negative_rate_constant": (
        _net([Reaction("r", {"A": 1}, {"B": 1}, MA(-1.0))]),
        ErrorCode.INFEASIBLE_RATE,
    ),
    "nan_rate_constant": (
        _net([Reaction("r", {"A": 1}, {"B": 1}, MA(float("nan")))]),
        ErrorCode.INFEASIBLE_RATE,
    ),
    "unknown_rate_law": (
        _net([Reaction("r", {"A": 1}, {"B": 1}, RateLaw("hill", {"k": 1.0}))]),
        ErrorCode.INFEASIBLE_RATE,
    ),
    "missing_rate_param": (
        _net([Reaction("r", {"A": 1}, {"B": 1}, RateLaw("mass_action", {}))]),
        ErrorCode.INFEASIBLE_RATE,
    ),
    "source_in_closed_network": (
        _net([Reaction("r", {}, {"A": 1}, MA(1.0))]),
        ErrorCode.UNCLOSED_NETWORK,
    ),
    "sink_in_closed_network": (
        _net([Reaction("r", {"A": 1}, {}, MA(1.0))]),
        ErrorCode.UNCLOSED_NETWORK,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_invalid_networks_raise_distinct_codes(name):
    network, expected = CASES[name]
    log.info("input network: %s", network)
    log.info("expected code: %s", expected.value)
    with pytest.raises(NetworkValidationError) as excinfo:
        validate_network(network)
    log.info("got code=%s detail=%s", excinfo.value.code, excinfo.value.detail)
    assert excinfo.value.code == expected


def _payload(net: Network) -> dict:
    return {
        "species": net.species,
        "boundary_species": sorted(net.boundary_species),
        "reactions": [
            {
                "name": r.name,
                "reactants": r.reactants,
                "products": r.products,
                "rate": {"kind": r.rate.kind, "params": r.rate.params},
            }
            for r in net.reactions
        ],
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_api_and_offline_validation_agree(name):
    """The /validate endpoint must reach the same verdict as validate_network."""
    network, expected = CASES[name]
    client = TestClient(create_app())
    # json.dumps (allow_nan=True) is needed to transport NaN/Infinity
    resp = client.post(
        "/validate",
        content=json.dumps(_payload(network)),
        headers={"content-type": "application/json"},
    )
    log.info("case=%s http_status=%s body=%s", name, resp.status_code, resp.json())
    assert resp.status_code == 422
    assert resp.json()["code"] == expected.value
    with pytest.raises(NetworkValidationError) as excinfo:
        validate_network(network)
    assert excinfo.value.code.value == resp.json()["code"]


def test_open_network_with_source_is_valid():
    net = _net([Reaction("feed", {}, {"A": 1}, MA(0.5))], boundary=("A",))
    validate_network(net)
    log.info("open network with boundary species accepted: %s", net)
