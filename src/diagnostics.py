"""Persistence for optional training diagnostics."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

from src.regression import TrainingResult


class DiagnosticsError(ValueError):
    """Raised when training diagnostics cannot be written."""


def save_training_diagnostics(
    result: TrainingResult,
    learning_rate: float,
    iterations: int,
    path: str | Path,
) -> None:
    """Write training configuration and results as an atomic JSON document."""
    _validate_finite(learning_rate, "learning_rate")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise DiagnosticsError("Invalid iterations: expected a positive integer")
    if iterations <= 0:
        raise DiagnosticsError("Invalid iterations: expected a positive integer")
    if not result.cost_history:
        raise DiagnosticsError("Invalid training result: cost history is empty")
    if len(result.cost_history) != iterations + 1:
        raise DiagnosticsError(
            "Invalid training result: cost history length does not match iterations"
        )

    numeric_values = (
        result.theta0,
        result.theta1,
        result.normalized_theta0,
        result.normalized_theta1,
        result.mean_km,
        result.std_km,
        *result.cost_history,
    )
    for value in numeric_values:
        _validate_finite(value, "training result")

    payload = {
        "learning_rate": learning_rate,
        "iterations": iterations,
        "theta0": result.theta0,
        "theta1": result.theta1,
        "normalized_theta0": result.normalized_theta0,
        "normalized_theta1": result.normalized_theta1,
        "mean_km": result.mean_km,
        "std_km": result.std_km,
        "initial_cost": result.cost_history[0],
        "final_cost": result.cost_history[-1],
        "cost_history": list(result.cost_history),
    }
    diagnostics_path = Path(path)
    temporary_path: Path | None = None

    try:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=diagnostics_path.parent,
            prefix=f".{diagnostics_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, diagnostics_path)
        temporary_path = None
    except (OSError, TypeError, ValueError) as error:
        raise DiagnosticsError(
            f"Cannot save diagnostics '{diagnostics_path}': {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _validate_finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticsError(f"Invalid {name}: expected a finite number")
    if not math.isfinite(value):
        raise DiagnosticsError(f"Invalid {name}: expected a finite number")
