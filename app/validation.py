"""Network validation: single entry point reused by API and batch scans.

Both the HTTP interface and offline/batch scanning call
:func:`validate_network`, so identical networks always yield identical
verdicts. Nothing is silently corrected: any violation raises
:class:`NetworkValidationError` with a distinguishing ``code``.
"""

from __future__ import annotations

import math

from .errors import ErrorCode, NetworkValidationError
from .models import Network, RateLaw, Reaction

#: rate-law kinds understood by the simulator, with required parameters
_RATE_LAW_PARAMS: dict[str, frozenset[str]] = {
    "mass_action": frozenset({"k"}),
    "michaelis_menten": frozenset({"kcat", "Km", "enzyme"}),
}


def _check_rate_law(reaction_name: str, law: RateLaw, direction: str) -> None:
    if law.kind not in _RATE_LAW_PARAMS:
        raise NetworkValidationError(
            ErrorCode.INFEASIBLE_RATE,
            f"reaction '{reaction_name}' {direction}: unknown rate law "
            f"kind '{law.kind}' (expected one of {sorted(_RATE_LAW_PARAMS)})",
        )
    required = _RATE_LAW_PARAMS[law.kind]
    missing = required - law.params.keys()
    if missing:
        raise NetworkValidationError(
            ErrorCode.INFEASIBLE_RATE,
            f"reaction '{reaction_name}' {direction}: rate law "
            f"'{law.kind}' is missing parameters {sorted(missing)}",
        )
    for key, value in law.params.items():
        if key == "enzyme":
            if not isinstance(value, str):
                raise NetworkValidationError(
                    ErrorCode.INFEASIBLE_RATE,
                    f"reaction '{reaction_name}' {direction}: 'enzyme' "
                    f"must name a species, got {value!r}",
                )
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise NetworkValidationError(
                ErrorCode.INFEASIBLE_RATE,
                f"reaction '{reaction_name}' {direction}: parameter "
                f"'{key}' must be a finite non-negative number, got {value!r}",
            )


def _check_reaction(network: Network, reaction: Reaction) -> None:
    known = set(network.species)
    for side_name, side in (("reactants", reaction.reactants), ("products", reaction.products)):
        for species, coeff in side.items():
            if species not in known:
                raise NetworkValidationError(
                    ErrorCode.MISSING_SPECIES,
                    f"reaction '{reaction.name}' references unknown species "
                    f"'{species}' in {side_name}",
                )
            if not isinstance(coeff, (int, float)) or not math.isfinite(coeff) or coeff <= 0:
                raise NetworkValidationError(
                    ErrorCode.ILLEGAL_STOICHIOMETRY,
                    f"reaction '{reaction.name}' has illegal stoichiometric "
                    f"coefficient {coeff!r} for species '{species}' "
                    f"(must be a finite positive number)",
                )
    if not reaction.reactants and not reaction.products:
        raise NetworkValidationError(
            ErrorCode.ILLEGAL_STOICHIOMETRY,
            f"reaction '{reaction.name}' has neither reactants nor products",
        )
    if network.closed and (not reaction.reactants or not reaction.products):
        raise NetworkValidationError(
            ErrorCode.UNCLOSED_NETWORK,
            f"reaction '{reaction.name}' is a "
            f"{'source' if not reaction.reactants else 'sink'} but network "
            f"is declared closed (no boundary species)",
        )
    _check_rate_law(reaction.name, reaction.rate, "forward")
    if reaction.reverse is not None:
        _check_rate_law(reaction.name, reaction.reverse, "reverse")
        # enzyme of a Michaelis-Menten law must be a known species
    for law in (reaction.rate, reaction.reverse):
        if law is not None and law.kind == "michaelis_menten":
            enzyme = law.params["enzyme"]
            if enzyme not in known:
                raise NetworkValidationError(
                    ErrorCode.MISSING_SPECIES,
                    f"reaction '{reaction.name}' uses unknown enzyme species "
                    f"'{enzyme}'",
                )


def validate_network(network: Network) -> None:
    """Validate a network, raising NetworkValidationError on any problem.

    Checks (each mapped to a distinct ErrorCode):
      - unknown species references (incl. enzymes) -> MISSING_SPECIES
      - non-positive / non-finite stoichiometry    -> ILLEGAL_STOICHIOMETRY
      - negative / non-finite / incomplete rates   -> INFEASIBLE_RATE
      - source/sink reaction in a closed network   -> UNCLOSED_NETWORK
    """
    if not network.species:
        raise NetworkValidationError(
            ErrorCode.MISSING_SPECIES, "network declares no species"
        )
    for name, conc in network.species.items():
        if not isinstance(conc, (int, float)) or not math.isfinite(conc) or conc < 0:
            raise NetworkValidationError(
                ErrorCode.ILLEGAL_STOICHIOMETRY,
                f"species '{name}' has illegal initial concentration "
                f"{conc!r} (must be a finite non-negative number)",
            )
    for boundary in network.boundary_species:
        if boundary not in network.species:
            raise NetworkValidationError(
                ErrorCode.MISSING_SPECIES,
                f"boundary species '{boundary}' is not declared in species",
            )
    for reaction in network.reactions:
        _check_reaction(network, reaction)
