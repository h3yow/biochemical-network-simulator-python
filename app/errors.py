"""Error taxonomy for network validation and simulation.

Every failure carries a machine-readable ``code`` so that distinct causes
(missing species, illegal stoichiometry, infeasible rate, unclosed network)
remain distinguishable and are never silently corrected or swallowed.
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    MISSING_SPECIES = "MISSING_SPECIES"
    ILLEGAL_STOICHIOMETRY = "ILLEGAL_STOICHIOMETRY"
    INFEASIBLE_RATE = "INFEASIBLE_RATE"
    UNCLOSED_NETWORK = "UNCLOSED_NETWORK"
    DUPLICATE_SPECIES = "DUPLICATE_SPECIES"
    UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
    CONSERVATION_VIOLATION = "CONSERVATION_VIOLATION"
    NEGATIVE_CONCENTRATION = "NEGATIVE_CONCENTRATION"
    INVALID_COMPOSITION = "INVALID_COMPOSITION"
    UNBALANCED_REACTION = "UNBALANCED_REACTION"


class NetworkError(Exception):
    """Base error carrying a stable code and a human-readable detail."""

    def __init__(self, code: ErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class NetworkValidationError(NetworkError):
    """Raised when a network description fails validation."""


class SimulationError(NetworkError):
    """Raised when a simulation violates a physical invariant."""
