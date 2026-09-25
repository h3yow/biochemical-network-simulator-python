"""Parse raw JSON-like dicts (API payloads) into domain models.

The parser performs *structural* validation only (required fields, types,
finite/feasible scalar values). Cross-reference and closure checks live in
:mod:`app.validation` so that every entry point shares exactly one semantic
validation pass.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import (
    InvalidInitialStateError,
    InvalidRateError,
    InvalidStoichiometryError,
    NetworkValidationError,
    SimConfigError,
)
from .models import Network, RateLaw, RateLawKind, Reaction, Species


def _require_mapping(obj: Any, context: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise NetworkValidationError(
            f"{context} must be an object", details={"context": context}
        )
    return obj


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NetworkValidationError(
            f"{context} must be a number", details={"context": context, "value": value}
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise NetworkValidationError(
            f"{context} must be finite", details={"context": context, "value": value}
        )
    return fv


def _nonempty_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise NetworkValidationError(
            f"{context} must be a non-empty string",
            details={"context": context, "value": value},
        )
    return value


def _parse_side(side: Any, context: str) -> dict[str, float]:
    """Parse a stoichiometric side, e.g. {'S': 1, 'E': 1}."""
    if side is None:
        return {}
    side = _require_mapping(side, context)
    result: dict[str, float] = {}
    for sp, coef in side.items():
        sp = _nonempty_str(sp, f"{context} species id")
        if isinstance(coef, bool) or not isinstance(coef, (int, float)):
            raise InvalidStoichiometryError(
                f"stoichiometric coefficient for '{sp}' in {context} must be a number",
                details={"species": sp, "coefficient": coef},
            )
        cf = float(coef)
        if not math.isfinite(cf) or cf < 0:
            raise InvalidStoichiometryError(
                f"stoichiometric coefficient for '{sp}' in {context} must be a "
                "finite non-negative number",
                details={"species": sp, "coefficient": coef},
            )
        if cf == 0:
            raise InvalidStoichiometryError(
                f"stoichiometric coefficient for '{sp}' in {context} must be > 0 "
                "(remove the species instead of writing 0)",
                details={"species": sp, "coefficient": coef},
            )
        result[sp] = cf
    return result


def _parse_rate_law(rl: Any, context: str) -> RateLaw:
    rl = _require_mapping(rl, context)
    kind_raw = _nonempty_str(rl.get("kind"), f"{context}.kind")
    try:
        kind = RateLawKind(kind_raw)
    except ValueError:
        raise InvalidRateError(
            f"unknown rate law kind '{kind_raw}' in {context}",
            details={"context": context, "kind": kind_raw,
                     "allowed": [k.value for k in RateLawKind]},
        )
    if "k" not in rl:
        raise InvalidRateError(
            f"rate constant 'k' is missing in {context}",
            details={"context": context},
        )
    k = _finite_number(rl["k"], f"{context}.k")
    if k < 0:
        raise InvalidRateError(
            f"rate constant 'k' in {context} must be >= 0",
            details={"context": context, "k": k},
        )
    if kind is RateLawKind.ENZYMATIC:
        enzyme = rl.get("enzyme")
        if not isinstance(enzyme, str) or not enzyme:
            raise InvalidRateError(
                f"enzymatic rate law in {context} requires a non-empty 'enzyme' "
                "species id",
                details={"context": context},
            )
        if "km" not in rl:
            raise InvalidRateError(
                f"enzymatic rate law in {context} requires 'km'",
                details={"context": context, "enzyme": enzyme},
            )
        km = _finite_number(rl["km"], f"{context}.km")
        if km <= 0:
            raise InvalidRateError(
                f"Michaelis constant 'km' in {context} must be > 0",
                details={"context": context, "km": km},
            )
        return RateLaw(kind=kind, k=k, enzyme=enzyme, km=km)
    return RateLaw(kind=kind, k=k)


def _parse_species(sp: Any, index: int) -> Species:
    sp = _require_mapping(sp, f"species[{index}]")
    sid = _nonempty_str(sp.get("id"), f"species[{index}].id")
    if "initial_concentration" not in sp:
        raise InvalidInitialStateError(
            f"species '{sid}' is missing initial_concentration",
            details={"species": sid},
        )
    c0 = _finite_number(sp["initial_concentration"],
                        f"species[{index}].initial_concentration")
    if c0 < 0:
        raise InvalidInitialStateError(
            f"initial concentration of '{sid}' must be >= 0",
            details={"species": sid, "initial_concentration": c0},
        )
    comp = sp.get("composition")
    composition: dict[str, float] | None = None
    if comp is not None:
        comp = _require_mapping(comp, f"species['{sid}'].composition")
        composition = {}
        for atom, amount in comp.items():
            atom = _nonempty_str(atom, f"species['{sid}'].composition atom id")
            av = _finite_number(amount, f"species['{sid}'].composition['{atom}']")
            if av <= 0:
                raise InvalidStoichiometryError(
                    f"composition of '{sid}' must use positive amounts",
                    details={"species": sid, "atom": atom, "amount": av},
                )
            composition[atom] = av
    return Species(id=sid, initial_concentration=c0, composition=composition)


def _parse_reaction(rx: Any, index: int) -> Reaction:
    rx = _require_mapping(rx, f"reaction[{index}]")
    rid = _nonempty_str(rx.get("id"), f"reaction[{index}].id")
    reactants = _parse_side(rx.get("reactants"), f"reaction['{rid}'].reactants")
    products = _parse_side(rx.get("products"), f"reaction['{rid}'].products")
    if "forward" not in rx:
        raise InvalidRateError(
            f"reaction '{rid}' is missing a 'forward' rate law",
            details={"reaction": rid},
        )
    forward = _parse_rate_law(rx["forward"], f"reaction['{rid}'].forward")
    reversible = bool(rx.get("reversible", False))
    reverse = None
    if reversible:
        if "reverse" not in rx:
            raise InvalidRateError(
                f"reversible reaction '{rid}' must declare a 'reverse' rate law",
                details={"reaction": rid},
            )
        reverse = _parse_rate_law(rx["reverse"], f"reaction['{rid}'].reverse")
    elif "reverse" in rx:
        raise InvalidRateError(
            f"reaction '{rid}' declares a reverse rate law but is not reversible",
            details={"reaction": rid},
        )
    return Reaction(
        id=rid, reactants=reactants, products=products,
        forward=forward, reversible=reversible, reverse=reverse,
    )


def parse_network(payload: dict[str, Any]) -> Network:
    """Convert a JSON-compatible payload into a :class:`Network`.

    Raises a :class:`~app.errors.NetworkValidationError` subclass on any
    structural problem. Semantic cross-checks follow in validation.
    """
    payload = _require_mapping(payload, "network")
    model_id = _nonempty_str(payload.get("model_id"), "model_id")
    version = payload.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise NetworkValidationError(
            "version must be a positive integer", details={"version": version}
        )
    species_raw = payload.get("species")
    if not isinstance(species_raw, list) or not species_raw:
        raise NetworkValidationError(
            "network must declare a non-empty species list",
            details={"species": species_raw},
        )
    species = tuple(_parse_species(sp, i) for i, sp in enumerate(species_raw))
    seen: set[str] = set()
    for sp in species:
        if sp.id in seen:
            raise NetworkValidationError(
                f"duplicate species id '{sp.id}'", details={"species": sp.id}
            )
        seen.add(sp.id)

    reactions_raw = payload.get("reactions")
    if not isinstance(reactions_raw, list) or not reactions_raw:
        raise NetworkValidationError(
            "network must declare a non-empty reaction list",
            details={"reactions": reactions_raw},
        )
    reactions = tuple(_parse_reaction(rx, i) for i, rx in enumerate(reactions_raw))
    seen_r: set[str] = set()
    for rx in reactions:
        if rx.id in seen_r:
            raise NetworkValidationError(
                f"duplicate reaction id '{rx.id}'", details={"reaction": rx.id}
            )
        seen_r.add(rx.id)

    return Network(model_id=model_id, version=version,
                   species=species, reactions=reactions)


def parse_sim_config(payload: dict[str, Any]) -> "SimConfig":
    """Parse a simulation config payload; defaults follow the documented tolerances."""
    from .models import SimConfig

    payload = _require_mapping(payload, "config")
    for key in ("t_end", "output_step"):
        if key not in payload:
            raise SimConfigError(
                f"config is missing '{key}'", details={"key": key}
            )
    t_end = _finite_number(payload["t_end"], "config.t_end")
    output_step = _finite_number(payload["output_step"], "config.output_step")
    if t_end <= 0 or output_step <= 0:
        raise SimConfigError(
            "config t_end and output_step must be > 0",
            details={"t_end": t_end, "output_step": output_step},
        )
    rel_tol = _finite_number(payload.get("rel_tol", 1e-6), "config.rel_tol")
    abs_tol = _finite_number(payload.get("abs_tol", 1e-9), "config.abs_tol")
    conservation_tol = _finite_number(
        payload.get("conservation_tol", 1e-6), "config.conservation_tol")
    negative_conc_tol = _finite_number(
        payload.get("negative_conc_tol", 1e-10), "config.negative_conc_tol")
    for name, v in (("rel_tol", rel_tol), ("abs_tol", abs_tol),
                    ("conservation_tol", conservation_tol),
                    ("negative_conc_tol", negative_conc_tol)):
        if v <= 0:
            raise SimConfigError(
                f"config {name} must be > 0", details={name: v}
            )
    return SimConfig(t_end=t_end, output_step=output_step, rel_tol=rel_tol,
                     abs_tol=abs_tol, conservation_tol=conservation_tol,
                     negative_conc_tol=negative_conc_tol)
