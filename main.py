"""FastAPI entry point for the biochemical network simulation service."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.errors import SimError
from app.parser import parse_sim_config
from app.serialize import result_to_dict, scan_point_to_dict
from app.service import SimulationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("bionet.api")

app = FastAPI(title="Biochemical Network Simulator")

# Module-level service: an append-only immutable registry plus a lock; every
# run/scan builds its own copied network state (see app/service.py).
service = SimulationService()


@app.exception_handler(SimError)
async def sim_error_handler(request: Request, exc: SimError) -> JSONResponse:
    # Distinct stable code per failure cause; details aid locating the reason.
    logger.warning("request failed path=%s code=%s message=%s details=%s",
                   request.url.path, exc.code, exc.message, exc.details)
    return JSONResponse(
        status_code=400 if exc.code != "unknown_model" else 404,
        content={"error": {"code": exc.code, "message": exc.message,
                           "details": exc.details}},
    )


class RegisterRequest(BaseModel):
    network: dict[str, Any]


class RunRequest(BaseModel):
    model_id: str
    version: int | None = None
    config: dict[str, Any]
    overrides: dict[str, float] | None = None


class ScanRequest(BaseModel):
    model_id: str
    version: int | None = None
    config: dict[str, Any]
    parameter_path: str
    values: list[float]


@app.post("/networks")
def register_network(req: RegisterRequest) -> dict:
    return service.register_network(req.network)


@app.post("/simulate")
def run_simulation(req: RunRequest) -> dict:
    cfg = parse_sim_config(req.config)
    result = service.run(req.model_id, req.version, cfg, req.overrides)
    return result_to_dict(result)


@app.post("/scan")
def run_scan(req: ScanRequest) -> dict:
    cfg = parse_sim_config(req.config)
    points = service.scan(req.model_id, req.version, cfg,
                          req.parameter_path, req.values)
    return {"model_id": req.model_id,
            "parameter_path": req.parameter_path,
            "points": [scan_point_to_dict(p) for p in points]}


@app.get("/")
def read_root() -> dict:
    return {"service": "biochemical-network-simulator", "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
