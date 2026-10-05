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

#: relative tolerance for element-balance checks on stoichiometry
_BALANCE_TOL = 1e-9


def _check_composition_declarations(network: Network) -> None:
    """Validate composition declarations (all-or-nothing, positive counts).

    Skipped entirely for legacy networks without any composition, so their
    verdicts are unchanged. When compositions ARE declared, every declared
    species must carry one and every count must be a finite positive number.
    """
    comps = network.compositions
    if not comps:
        return
    for name in comps:
        if name not in network.species:
            raise NetworkValidationError(
                ErrorCode.MISSING_SPECIES,
                f"composition declared for unknown species '{name}'",
            )
    for name in network.species:
        if name not in comps:
            raise NetworkValidationError(
                ErrorCode.INVALID_COMPOSITION,
                f"species '{name}' has no composition while other species "
                f"do (compositions are all-or-nothing)",
            )
    for name, comp in comps.items():
        if not comp:
            raise NetworkValidationError(
                ErrorCode.INVALID_COMPOSITION,
                f"species '{name}' declares an empty composition",
            )
        for element, count in comp.items():
            if not isinstance(element, str) or not element:
                raise NetworkValidationError(
                    ErrorCode.INVALID_COMPOSITION,
                    f"species '{name}' has invalid element name {element!r}",
                )
            if not isinstance(count, (int, float)) or not math.isfinite(count) or count <= 0:
                raise NetworkValidationError(
                    ErrorCode.INVALID_COMPOSITION,
                    f"species '{name}' element '{element}' count must be a "
                    f"finite positive number, got {count!r}",
                )


def _check_element_balance(network: Network) -> None:
    """Check every reaction conserves each element (forward and reverse
    share the same stoichiometry, so one check covers both directions).
    Any imbalance is reported with the exact reaction, element and both
    sides' totals; nothing is silently corrected."""
    comps = network.compositions
    if not comps:
        return
    elements = sorted({e for comp in comps.values() for e in comp})
    for reaction in network.reactions:
        for element in elements:
            lhs = sum(
                coeff * comps[species].get(element, 0.0)
                for species, coeff in reaction.reactants.items()
            )
            rhs = sum(
                coeff * comps[species].get(element, 0.0)
                for species, coeff in reaction.products.items()
            )
            limit = _BALANCE_TOL * max(abs(lhs), abs(rhs), 1.0)
            if abs(lhs - rhs) > limit:
                raise NetworkValidationError(
                    ErrorCode.UNBALANCED_REACTION,
                    f"reaction '{reaction.name}' does not conserve element "
                    f"'{element}': reactants supply {lhs!r}, products "
                    f"{rhs!r} (|diff|={abs(lhs - rhs):.3e} > {limit:.1e})",
                )


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
      - illegal/contradictory compositions         -> INVALID_COMPOSITION
      - element imbalance in a reaction            -> UNBALANCED_REACTION
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
    _check_composition_declarations(network)
    for boundary in network.boundary_species:
        if boundary not in network.species:
            raise NetworkValidationError(
                ErrorCode.MISSING_SPECIES,
                f"boundary species '{boundary}' is not declared in species",
            )
    for reaction in network.reactions:
        _check_reaction(network, reaction)
    _check_element_balance(network)
