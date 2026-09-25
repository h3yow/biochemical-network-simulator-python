"""Convert domain results to JSON-safe payloads."""

from __future__ import annotations

from .models import ScanPoint, SimResult


def result_to_dict(r: SimResult) -> dict:
    return {
        "model_id": r.model_id,
        "version": r.version,
        "times": list(r.times),
        "species_order": list(r.species_order),
        "values": [list(row) for row in r.values],
        "conserved_total": r.conserved_total,
        "max_conservation_error": r.max_conservation_error,
        "min_concentration": r.min_concentration,
        "steady_state": r.steady_state,
    }


def scan_point_to_dict(p: ScanPoint) -> dict:
    return {
        "parameter_path": p.parameter_path,
        "value": p.value,
        "result": result_to_dict(p.result) if p.result is not None else None,
        "error_code": p.error_code,
        "error_message": p.error_message,
        "ok": p.result is not None,
    }
