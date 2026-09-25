"""Error taxonomy for network validation and simulation.

Every failure mode has a distinct, stable ``code`` so callers (and tests)
can distinguish error *causes* rather than matching on free text.
"""

from __future__ import annotations


class SimError(Exception):
    """Base class for all service errors."""

    code: str = "sim_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NetworkValidationError(SimError):
    """Base class for network-description validation failures."""

    code = "network_invalid"


class MissingSpeciesError(NetworkValidationError):
    """A reaction references a species that is not declared."""

    code = "missing_species"


class InvalidStoichiometryError(NetworkValidationError):
    """A stoichiometric coefficient is missing, non-numeric or negative."""

    code = "invalid_stoichiometry"


class InvalidRateError(NetworkValidationError):
    """A rate law / rate constant is missing, malformed or infeasible."""

    code = "invalid_rate"


class UnclosedNetworkError(NetworkValidationError):
    """The network is not closed: atoms/species are created or destroyed."""

    code = "unclosed_network"


class InvalidInitialStateError(NetworkValidationError):
    """An initial concentration is missing, negative or non-finite."""

    code = "invalid_initial_state"


class NonPhysicalResultError(SimError):
    """Integration produced a negative concentration."""

    code = "non_physical_result"


class ConservationViolationError(SimError):
    """Total conserved quantity drifted beyond the configured tolerance."""

    code = "conservation_violation"


class IntegrationError(SimError):
    """The integrator failed to converge to the required accuracy."""

    code = "integration_failed"


class UnknownModelError(SimError):
    """A request referenced a model id/version that is not registered."""

    code = "unknown_model"


class SimConfigError(SimError):
    """A simulation configuration is missing fields or infeasible."""

    code = "invalid_config"
