"""Species composition: declarations, reaction balance and the network's
own conservation quantities.

A network may declare, per species, how many units of each atom/group it
contains. Those declarations make it possible to:

  * verify that every reaction (forward and reverse) is atom-balanced,
    reporting the exact reaction, direction and component on mismatch;
  * report each independent conservation quantity of the network
    (weighted sums of concentrations that stay constant) and the numeric
    drift of each one along a trajectory.

Networks without composition retain the legacy structural treatment.
All arithmetic on stoichiometry and composition counts is exact
(rational), so balance verdicts never depend on floating-point noise;
only *trajectory drift* is a floating-point tolerance question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

from .errors import ErrorCode, NetworkValidationError
from .models import Network, Reaction


@dataclass(frozen=True)
class ConservationLaw:
    """One independent conserved weighted sum: Σ weights[i]·[species_i].

    ``origin`` records how the law was obtained: ``"composition:<name>"``
    for an atom/group conservation law read off the composition table, or
    ``"structural:<n>"`` for a basis law of the stoichiometric left null
    space (the only kind available for legacy networks).
    """

    name: str
    weights: dict[str, int]
    origin: str


def validate_composition(network: Network) -> None:
    """Validate composition declarations; raise INVALID_COMPOSITION.

    Rules (all-or-nothing: a network either carries composition for
    every species or for none):
      * every key must name a declared species and vice versa;
      * each species maps component name -> positive finite count;
      * component names must be non-empty strings, counts positive
        finite numbers (integers in the common case); an empty
        component map is rejected as a useless declaration.
    A network with no composition at all is a legacy network and is
    accepted untouched.
    """
    comp = network.composition
    if not comp:
        return
    declared = set(network.species)
    if not isinstance(comp, dict):
        raise NetworkValidationError(
            ErrorCode.INVALID_COMPOSITION,
            "composition must be an object mapping species to "
            "{component: count}",
        )
    for species, entry in comp.items():
        if species not in declared:
            raise NetworkValidationError(
                ErrorCode.INVALID_COMPOSITION,
                f"composition declared for species '{species}' which is "
                f"not present in the species list",
            )
        if not isinstance(entry, dict) or not entry:
            raise NetworkValidationError(
                ErrorCode.INVALID_COMPOSITION,
                f"species '{species}' has an empty or non-object "
                f"composition declaration (expected {{component: count}} "
                f"with at least one component)",
            )
        for component, count in entry.items():
            if not isinstance(component, str) or not component:
                raise NetworkValidationError(
                    ErrorCode.INVALID_COMPOSITION,
                    f"species '{species}' has an invalid component name "
                    f"{component!r} (must be a non-empty string)",
                )
            if (
                not isinstance(count, (int, float))
                or isinstance(count, bool)
                or not math.isfinite(count)
                or count <= 0
            ):
                raise NetworkValidationError(
                    ErrorCode.INVALID_COMPOSITION,
                    f"species '{species}' component '{component}' has "
                    f"illegal count {count!r} (must be a finite positive "
                    f"number)",
                )
    missing = sorted(declared - set(comp))
    if missing:
        raise NetworkValidationError(
            ErrorCode.INVALID_COMPOSITION,
            f"species {missing} have no composition declaration: when a "
            f"network carries composition, every species must declare "
            f"its components",
        )


def _frac(value) -> Fraction:
    """Exact rational conversion from int/float JSON numbers.

    ``str(float)`` is used so that 0.1 becomes 1/10 rather than the
    binary float 3602879701896397/36028797018963968.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def _fmt_num(value: Fraction) -> str:
    """Human-readable exact number: integer without denominator."""
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def check_reaction_balance(network: Network, reaction: Reaction) -> None:
    """Check one reaction is atom-balanced; raise UNBALANCED_REACTION.

    Both directions are checked. Source (empty reactants) and sink
    (empty products) reactions exchange matter with the environment and
    only occur in open networks (boundary species present), so they are
    exempt from atom balance by definition; every genuine reaction with
    species on both sides must balance exactly. The detail lists, per
    component, the exact totals on both sides and the per-species
    contributions of the mismatched side, so the offending composition
    entry is pinpointed. Nothing is ever corrected: any mismatch raises.
    """
    comp = network.composition
    if not comp:
        return
    # source/sink reactions are exchange with the environment, not
    # internal transformations: they cannot be atom-balanced and need
    # not be (closed networks already forbid them at validation)
    if not reaction.reactants or not reaction.products:
        return

    def side_balance(side: dict[str, float]) -> dict[str, Fraction]:
        totals: dict[str, Fraction] = {}
        for species, coeff in side.items():
            for component, count in comp[species].items():
                totals[component] = (
                    totals.get(component, Fraction(0))
                    + _frac(coeff) * _frac(count)
                )
        return totals

    def _contrib(side, component: str) -> str:
        parts = []
        for species in sorted(side):
            count = comp[species].get(component)
            if count:
                parts.append(
                    f"{species}({_fmt_num(_frac(side[species]) * _frac(count))})"
                )
        return " + ".join(parts) if parts else "(none)"

    for direction, left_side, right_side in (
        ("forward", reaction.reactants, reaction.products),
        ("reverse", reaction.products, reaction.reactants),
    ):
        left = side_balance(left_side)
        right = side_balance(right_side)
        for component in sorted(set(left) | set(right)):
            lhs = left.get(component, Fraction(0))
            rhs = right.get(component, Fraction(0))
            if lhs != rhs:
                raise NetworkValidationError(
                    ErrorCode.UNBALANCED_REACTION,
                    f"reaction '{reaction.name}' {direction} is not "
                    f"balanced in component '{component}': reactant side "
                    f"has {_fmt_num(lhs)} [{_contrib(left_side, component)}], "
                    f"product side has {_fmt_num(rhs)} "
                    f"[{_contrib(right_side, component)}]",
                )


def _rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    """Reduced row echelon form over exact rationals; returns (matrix, pivots)."""
    m = [row[:] for row in matrix]
    rows = len(m)
    cols = len(m[0]) if rows else 0
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot = next((i for i in range(r, rows) if m[i][c] != 0), None)
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        pv = m[r][c]
        m[r] = [x / pv for x in m[r]]
        for i in range(rows):
            if i != r and m[i][c] != 0:
                factor = m[i][c]
                m[i] = [a - factor * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
    return m, pivots


def _primitive_int_vector(vec: list[Fraction]) -> list[int]:
    """Scale a rational vector into the smallest all-positive-leading
    integer vector: divide by the gcd of all entries and flip sign so the
    first nonzero entry is positive."""
    denoms = [x.denominator for x in vec]
    common = denoms[0]
    for d in denoms[1:]:
        common = common * d // gcd(common, d)
    ints = [int(x * common) for x in vec]
    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    if g > 0:
        ints = [v // g for v in ints]
    first = next(v for v in ints if v != 0)
    if first < 0:
        ints = [-v for v in ints]
    return ints


def _structural_nullspace(
    network: Network, species: tuple[str, ...]
) -> list[list[int]]:
    """Integer basis of the left null space {w : w · S = 0} of the
    stoichiometric matrix S (species x reactions), computed exactly."""
    idx = {name: i for i, name in enumerate(species)}
    n = len(species)
    reactions = tuple(sorted(network.reactions, key=lambda r: r.name))
    # S^T: one row per reaction, one column per species
    st = [[Fraction(0) for _ in range(n)] for _ in reactions]
    for j, reaction in enumerate(reactions):
        for name, coeff in reaction.products.items():
            st[j][idx[name]] += _frac(coeff)
        for name, coeff in reaction.reactants.items():
            st[j][idx[name]] -= _frac(coeff)
    rref, pivots = _rref(st)
    # free columns taken right-to-left yield the natural positive
    # primitive basis (for 2A+B->C: [2,1,1], not a signed variant)
    free_cols = [c for c in range(n - 1, -1, -1) if c not in pivots]
    basis: list[list[int]] = []
    for fc in free_cols:
        vec = [Fraction(0) for _ in range(n)]
        vec[fc] = Fraction(1)
        for row, pc in zip(rref, pivots):
            vec[pc] = -row[fc]
        basis.append(_primitive_int_vector(vec))
    return basis


def _merge_laws(
    structural: list[list[int]],
    components: list[tuple[str, list[Fraction]]],
) -> list[tuple[list[Fraction], str]]:
    """Build one deterministic maximal independent set of genuine
    conservation laws. Named composition vectors take the slots they
    span (and must lie in the stoichiometric left null space to be real
    invariants -- source/sink reactions in open networks can make a
    component non-conserved, in which case it is skipped); the
    remaining dimensions are filled by structural basis vectors. Each
    returned entry is (vector, origin), with origin
    ``"composition:<component>"`` or ``"structural"``."""
    kernel = [[Fraction(x) for x in v] for v in structural]

    def rank(vecs: list[list[Fraction]]) -> int:
        return len(_rref(vecs)[1])

    kernel_rank = rank(kernel)

    def in_kernel(vec: list[Fraction]) -> bool:
        return rank(kernel + [vec]) == kernel_rank

    chosen: list[list[Fraction]] = []
    tagged: list[tuple[list[Fraction], str]] = []

    def try_add(vec, origin) -> bool:
        if rank(chosen + [vec]) > len(chosen):
            chosen.append(vec)
            tagged.append((vec, origin))
            return True
        return False

    # named composition laws first, but only real invariants
    for name, vec in sorted(components, key=lambda kv: kv[0]):
        if in_kernel(vec):
            try_add(vec, f"composition:{name}")
    # fill the rest of the null space with structural basis vectors
    for vec in kernel:
        if len(chosen) == kernel_rank:
            break
        try_add(vec, "structural")
    return tagged


def conservation_laws(
    network: Network, species: tuple[str, ...]
) -> tuple[ConservationLaw, ...]:
    """Return the network's independent conservation laws.

    With composition, each atom/group contributes one named law and
    further structural laws (e.g. an unconsumed enzyme pool) are added;
    without composition the exact left-null-space basis is returned
    directly. Every law is an exact integer weighted sum of species.
    """
    structural = _structural_nullspace(network, species)
    if network.has_composition:
        component_names = sorted(
            {c for entry in network.composition.values() for c in entry}
        )
        components: list[tuple[str, list[Fraction]]] = []
        for cname in component_names:
            vec = [
                _frac(network.composition[s].get(cname, 0)) for s in species
            ]
            components.append((cname, vec))
        merged = _merge_laws(structural, components)
    else:
        merged = [([Fraction(x) for x in v], "structural") for v in structural]

    laws: list[ConservationLaw] = []
    struct_index = 0
    for vec, origin in merged:
        ints = _primitive_int_vector(vec)
        weights = {s: w for s, w in zip(species, ints) if w != 0}
        if origin == "structural":
            name = f"structural:{struct_index}"
            struct_index += 1
        else:
            name = origin  # "composition:<component>"
        laws.append(ConservationLaw(name=name, weights=weights, origin=origin))
    return tuple(laws)
