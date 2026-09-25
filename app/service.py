"""Stateless service facade: model registry, perturbation and batch scan.

Isolation guarantees
--------------------
* Registered networks are immutable frozen dataclasses held in an append-only
  registry; runs never mutate them.
* Each run / scan point works on an independent ``dataclasses.replace`` copy
  with the override applied. No mutable object is shared between requests,
  scan points or threads.
* No result is cached across parameter values or model versions; a result is
  always recomputed from its exact inputs, so different versions or rates can
  never be confused with each other. Determinism makes identical requests
  reproducible regardless.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from typing import Any

from .errors import SimError, UnknownModelError
from .models import Network, ScanPoint, SimConfig, SimResult
from .parser import parse_network, parse_sim_config
from .simulation import simulate
from .validation import validate_network

logger = logging.getLogger("bionet.service")


def _network_fingerprint(network: Network) -> tuple:
    return (
        tuple((s.id, s.initial_concentration,
               tuple(sorted((s.composition or {}).items())))
              for s in network.species),
        tuple((r.id, tuple(sorted(r.reactants.items())),
               tuple(sorted(r.products.items())), r.reversible,
               (r.forward.kind.value, r.forward.k, r.forward.enzyme, r.forward.km),
               (None if r.reverse is None else
                (r.reverse.kind.value, r.reverse.k, r.reverse.enzyme, r.reverse.km)))
              for r in network.reactions),
    )


class SimulationService:
    def __init__(self) -> None:
        self._registry: dict[str, dict[int, Network]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_network(self, payload: dict[str, Any]) -> dict:
        network = parse_network(payload)
        # Every entry point validates through the same function.
        validate_network(network)
        with self._lock:
            versions = self._registry.setdefault(network.model_id, {})
            existing = versions.get(network.version)
            if existing is not None:
                if _network_fingerprint(existing) != _network_fingerprint(network):
                    raise SimError(
                        f"model '{network.model_id}' version {network.version} "
                        "already exists with different content; register a new "
                        "version instead",
                        details={"model_id": network.model_id,
                                 "version": network.version},
                    )
                network = existing
            else:
                versions[network.version] = network
        logger.info(
            "registered network model_id=%s version=%d species=%d reactions=%d",
            network.model_id, network.version,
            len(network.species), len(network.reactions),
        )
        return {"model_id": network.model_id, "version": network.version,
                "species": len(network.species),
                "reactions": len(network.reactions),
                "versions": sorted(self._registry[network.model_id])}

    def _resolve(self, model_id: str, version: int | None) -> Network:
        versions = self._registry.get(model_id)
        if not versions:
            raise UnknownModelError(
                f"model '{model_id}' is not registered",
                details={"model_id": model_id},
            )
        if version is None:
            return versions[max(versions)]
        network = versions.get(version)
        if network is None:
            raise UnknownModelError(
                f"model '{model_id}' version {version} is not registered",
                details={"model_id": model_id, "version": version,
                         "available": sorted(versions)},
            )
        return network

    # ------------------------------------------------------------------
    # Parameter perturbation
    # ------------------------------------------------------------------

    def _apply_override(self, network: Network,
                        overrides: dict[str, float]) -> Network:
        """Return an independent network copy with overrides applied."""
        species = {s.id: s for s in network.species}
        reactions = {r.id: r for r in network.reactions}
        new_species = list(network.species)
        new_reactions = list(network.reactions)

        for path, value in overrides.items():
            parts = path.split(".")
            if len(parts) == 3 and parts[0] == "species" and parts[2] == \
                    "initial_concentration":
                sid = parts[1]
                if sid not in species:
                    raise SimError(
                        f"override path '{path}' refers to unknown species",
                        details={"path": path},
                    )
                i = next(i for i, s in enumerate(new_species) if s.id == sid)
                new_species[i] = replace(new_species[i],
                                         initial_concentration=float(value))
            elif (len(parts) == 4 and parts[0] == "reactions"
                  and parts[3] in ("k", "km")):
                _, rid, direction, attr = parts
                if rid not in reactions or direction not in ("forward", "reverse"):
                    raise SimError(
                        f"override path '{path}' refers to unknown reaction/direction",
                        details={"path": path},
                    )
                rx = reactions[rid]
                law = rx.forward if direction == "forward" else rx.reverse
                if law is None:
                    raise SimError(
                        f"override path '{path}' targets a missing rate law",
                        details={"path": path},
                    )
                i = next(i for i, r in enumerate(new_reactions) if r.id == rid)
                new_law = replace(law, **{attr: float(value)})
                if direction == "forward":
                    new_reactions[i] = replace(new_reactions[i], forward=new_law)
                else:
                    new_reactions[i] = replace(new_reactions[i], reverse=new_law)
            else:
                raise SimError(
                    f"unsupported override path '{path}'",
                    details={"path": path,
                             "format": "species.<id>.initial_concentration | "
                                       "reactions.<id>.forward|reverse.k|km"},
                )
        return replace(network, species=tuple(new_species),
                       reactions=tuple(new_reactions))

    def run(self, model_id: str, version: int | None, config: SimConfig,
            overrides: dict[str, float] | None = None) -> SimResult:
        base = self._resolve(model_id, version)
        effective = self._apply_override(base, overrides or {})
        # Re-validate the perturbed network through the identical pass.
        validate_network(effective)
        logger.info(
            "simulate model_id=%s version=%d overrides=%s t_end=%g step=%g",
            model_id, effective.version, overrides or {},
            config.t_end, config.output_step,
        )
        result = simulate(effective, config)
        logger.info(
            "simulate done model_id=%s version=%d drift=%.3e min_conc=%.3e "
            "steady_state=%s",
            model_id, effective.version,
            result.max_conservation_error, result.min_concentration,
            result.steady_state is not None,
        )
        return result

    # ------------------------------------------------------------------
    # Batch scan
    # ------------------------------------------------------------------

    def scan(self, model_id: str, version: int | None, config: SimConfig,
             parameter_path: str, values: list[float]) -> list[ScanPoint]:
        if not values:
            raise SimError("scan requires a non-empty values list",
                           details={"parameter_path": parameter_path})
        points: list[ScanPoint] = []
        for value in values:
            # Independent override copy + validation per point; one bad value
            # is reported locally, never silently corrected or skipped.
            try:
                result = self.run(model_id, version, config,
                                  {parameter_path: float(value)})
                points.append(ScanPoint(parameter_path=parameter_path,
                                        value=float(value), result=result))
            except SimError as exc:
                logger.info(
                    "scan point failed model_id=%s path=%s value=%g code=%s",
                    model_id, parameter_path, value, exc.code,
                )
                points.append(ScanPoint(parameter_path=parameter_path,
                                        value=float(value), result=None,
                                        error_code=exc.code,
                                        error_message=str(exc)))
        return points
