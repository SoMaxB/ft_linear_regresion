"""Shared validation helpers for finite numeric user input."""

from __future__ import annotations

import math


def parse_finite_float(value: str, field: str = "value") -> float:
    """Parse a string as a finite floating-point value."""
    try:
        number = float(value.strip())
    except ValueError as error:
        raise ValueError(f"{field} must be a number") from error

    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number
