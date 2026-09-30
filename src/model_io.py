"""Persistence for the original-scale linear regression parameters."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile


class ModelError(ValueError):
    """Raised when a model cannot be saved or loaded."""


def save_model(
    theta0: float,
    theta1: float,
    path: str | Path = "model/theta.json",
) -> None:
    """Save finite original-scale parameters as a JSON model file.

    The destination directory is created when necessary. The temporary file
    and replace step prevent a failed write from leaving a partial model.
    """
    _validate_parameter(theta0, "theta0")
    _validate_parameter(theta1, "theta1")
    model_path = Path(path)
    temporary_path: Path | None = None

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=model_path.parent,
            prefix=f".{model_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump({"theta0": theta0, "theta1": theta1}, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, model_path)
        temporary_path = None
    except (OSError, TypeError, ValueError) as error:
        raise ModelError(f"Cannot save model '{model_path}': {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def load_model(
    path: str | Path = "model/theta.json",
) -> tuple[float, float]:
    """Load and validate ``(theta0, theta1)`` from a JSON model file."""
    model_path = Path(path)
    try:
        with model_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError as error:
        raise ModelError(f"Model not found: {model_path}") from error
    except json.JSONDecodeError as error:
        raise ModelError(f"Invalid JSON in model '{model_path}': {error.msg}") from error
    except UnicodeDecodeError as error:
        raise ModelError(f"Model is not valid UTF-8: {model_path}") from error
    except OSError as error:
        raise ModelError(f"Cannot read model '{model_path}': {error}") from error

    if not isinstance(payload, dict):
        raise ModelError("Invalid model: expected a JSON object")
    if set(payload) != {"theta0", "theta1"}:
        raise ModelError("Invalid model: expected exactly 'theta0' and 'theta1'")

    _validate_parameter(payload["theta0"], "theta0")
    _validate_parameter(payload["theta1"], "theta1")
    return float(payload["theta0"]), float(payload["theta1"])


def _validate_parameter(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"Invalid {name}: expected a finite number")
    if not math.isfinite(value):
        raise ModelError(f"Invalid {name}: expected a finite number")
