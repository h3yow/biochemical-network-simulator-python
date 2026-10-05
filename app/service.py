"""FastAPI service exposing validation, simulation and scanning.

All endpoints funnel through :func:`app.validation.validate_network`
and :func:`app.simulation.simulate`, so the HTTP interface and offline
batch scanning reach identical verdicts for identical networks.
Errors are returned as ``{"code": ..., "detail": ...}`` with HTTP 422,
never silently corrected.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .errors import ErrorCode, NetworkError, NetworkValidationError
from .models import Network, RateLaw, Reaction
from .scanning import ScanPoint, safe_scan_parameter
from .simulation import network_fingerprint, simulate
from .validation import validate_network


class RateLawIn(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class ReactionIn(BaseModel):
    name: str
    reactants: dict[str, float] = Field(default_factory=dict)
    products: dict[str, float] = Field(default_factory=dict)
    rate: RateLawIn
    reverse: RateLawIn | None = None


class NetworkIn(BaseModel):
    species: dict[str, float]
    reactions: list[ReactionIn] = Field(default_factory=list)
    boundary_species: list[str] = Field(default_factory=list)
    composition: dict[str, dict[str, float]] | None = None
    version: str = "1"


class SimulateIn(BaseModel):
    network: NetworkIn
    t_end: float
    n_points: int = 101
    t_start: float = 0.0
    conservation_tol: float = 1e-6


class ScanIn(BaseModel):
    network: NetworkIn
    reaction: str
    param: str
    values: list[float]
    t_end: float
    n_points: int = 101


def _to_domain(net: NetworkIn) -> Network:
    # duplicate boundary entries collapse silently in a frozenset; reject
    if len(set(net.boundary_species)) != len(net.boundary_species):
        raise NetworkValidationError(
            ErrorCode.DUPLICATE_SPECIES, "duplicate boundary species entries"
        )
    return Network(
        species=dict(net.species),
        reactions=tuple(
            Reaction(
                name=r.name,
                reactants=dict(r.reactants),
                products=dict(r.products),
                rate=RateLaw(r.rate.kind, dict(r.rate.params)),
                reverse=None
                if r.reverse is None
                else RateLaw(r.reverse.kind, dict(r.reverse.params)),
            )
            for r in net.reactions
        ),
        boundary_species=frozenset(net.boundary_species),
        composition={} if net.composition is None else {
            species: dict(entry) for species, entry in net.composition.items()
        },
        version=net.version,
    )


def _trajectory_json(tr) -> dict[str, Any]:
    return {
        "t": tr.t.tolist(),
        "species": list(tr.species),
        "y": tr.y.tolist(),
        "conserved_total": tr.conserved_total,
        "max_conservation_deviation": tr.max_conservation_deviation,
        "min_concentration": tr.min_concentration,
        "conservation_laws": [
            {
                "name": d.name,
                "origin": d.origin,
                "weights": d.weights,
                "initial_value": d.initial_value,
                "max_deviation": d.max_deviation,
                "tolerance": d.tolerance,
                "within_tolerance": d.within_tolerance,
            }
            for d in tr.conservation_laws
        ],
    }


def create_app() -> FastAPI:
    app = FastAPI(title="biochemical-network-simulator")

    @app.exception_handler(NetworkError)
    def _network_error(_request, exc: NetworkError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"code": exc.code.value, "detail": exc.detail}
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/validate")
    def validate_endpoint(net: NetworkIn) -> dict[str, Any]:
        network = _to_domain(net)
        validate_network(network)
        return {"valid": True, "fingerprint": network_fingerprint(network)}

    @app.post("/simulate")
    def simulate_endpoint(req: SimulateIn) -> dict[str, Any]:
        network = _to_domain(req.network)
        tr = simulate(
            network,
            t_end=req.t_end,
            n_points=req.n_points,
            t_start=req.t_start,
            conservation_tol=req.conservation_tol,
        )
        return {
            "fingerprint": network_fingerprint(network),
            "trajectory": _trajectory_json(tr),
        }

    @app.post("/scan")
    def scan_endpoint(req: ScanIn) -> dict[str, Any]:
        try:
            network = _to_domain(req.network)
            points = safe_scan_parameter(
                network,
                reaction=req.reaction,
                param=req.param,
                values=req.values,
                t_end=req.t_end,
                n_points=req.n_points,
            )
        except NetworkError as exc:
            # the base network itself cannot be built/validated, or the
            # scanned reaction/parameter does not exist: every point fails
            # for the same locatable reason, reported per point rather
            # than as one request-level rejection
            points = [
                ScanPoint(value=value, ok=False,
                          error_code=exc.code.value, error_detail=exc.detail)
                for value in req.values
            ]
        results = []
        for point in points:
            if point.ok:
                results.append(
                    {
                        "value": point.value,
                        "ok": True,
                        "fingerprint": point.fingerprint,
                        "trajectory": _trajectory_json(point.trajectory),
                    }
                )
            else:
                results.append(
                    {
                        "value": point.value,
                        "ok": False,
                        "code": point.error_code,
                        "detail": point.error_detail,
                    }
                )
        return {"results": results}

    return app
