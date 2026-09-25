"""Core domain types: species, reactions, networks, requests/results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RateLawKind(str, Enum):
    """Supported kinetic rate laws for a single (directed) reaction."""

    MASS_ACTION = "mass_action"
    ENZYMATIC = "enzymatic"  # Michaelis-Menten form


@dataclass(frozen=True)
class Species:
    id: str
    initial_concentration: float
    composition: dict[str, float] | None = None  # atom id -> amount


@dataclass(frozen=True)
class RateLaw:
    kind: RateLawKind
    k: float                      # forward rate constant / k_cat
    enzyme: str | None = None     # species id, required for enzymatic
    km: float | None = None       # Michaelis constant, required for enzymatic


@dataclass(frozen=True)
class Reaction:
    id: str
    reactants: dict[str, float]   # species id -> stoichiometric coefficient
    products: dict[str, float]
    forward: RateLaw
    reversible: bool = False
    reverse: RateLaw | None = None  # required when reversible is True


@dataclass(frozen=True)
class Network:
    model_id: str
    version: int
    species: tuple[Species, ...]
    reactions: tuple[Reaction, ...]


@dataclass(frozen=True)
class SimConfig:
    t_end: float
    output_step: float
    rel_tol: float = 1e-6
    abs_tol: float = 1e-9
    conservation_tol: float = 1e-6
    negative_conc_tol: float = 1e-10
    max_step: float | None = None


@dataclass(frozen=True)
class TimePoint:
    t: float
    concentrations: dict[str, float]


@dataclass(frozen=True)
class SimResult:
    model_id: str
    version: int
    times: tuple[float, ...]
    species_order: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]  # rows per time point
    conserved_total: float
    max_conservation_error: float
    min_concentration: float
    steady_state: dict[str, float] | None


@dataclass(frozen=True)
class ScanPoint:
    parameter_path: str
    value: float
    result: SimResult | None
    error_code: str | None = None
    error_message: str | None = None
