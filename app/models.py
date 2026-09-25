"""Data model for reaction network descriptions (pure, immutable)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RateLaw:
    """Kinetic rate law.

    kind: "mass_action" (rate = k * prod(reactants^stoich)) or
          "michaelis_menten" (rate = kcat*E*S/(Km+S), single substrate).
    params: rate constants, must be finite and non-negative.
    """

    kind: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Reaction:
    """A (possibly reversible) reaction.

    reactants/products map species name -> stoichiometric coefficient.
    Reversible reactions carry a ``reverse`` rate law in addition to
    the forward ``rate``.
    """

    name: str
    reactants: dict[str, float]
    products: dict[str, float]
    rate: RateLaw
    reverse: RateLaw | None = None


@dataclass(frozen=True)
class Network:
    """A reaction network. ``closed=False`` marks boundary (chemostat)
    species whose concentration is held fixed; a network with no boundary
    species must conserve total mass."""

    species: dict[str, float]  # name -> initial concentration
    reactions: tuple[Reaction, ...]
    boundary_species: frozenset[str] = frozenset()
    version: str = "1"

    @property
    def closed(self) -> bool:
        return not self.boundary_species
