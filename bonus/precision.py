"""Calculate prediction errors for a persisted regression model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
from pathlib import Path
import sys

# Make direct execution (`python3 bonus/precision.py`) behave like module execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DatasetError, load_dataset
from src.model_io import ModelError, load_model
from src.regression import estimate_price


DEFAULT_DATA = "data.csv"
DEFAULT_MODEL = "model/theta.json"


def calculate_mae(
    mileages: Sequence[float],
    prices: Sequence[float],
    theta0: float,
    theta1: float,
) -> float:
    """Return mean absolute error in the same units as the prices."""
    absolute_errors, _ = _calculate_errors(mileages, prices, theta0, theta1)
    mae = sum(absolute_errors) / len(absolute_errors)
    _ensure_finite(mae, "MAE")
    return mae


def calculate_mse(
    mileages: Sequence[float],
    prices: Sequence[float],
    theta0: float,
    theta1: float,
) -> float:
    """Return mean squared error in squared price units."""
    _, squared_errors = _calculate_errors(mileages, prices, theta0, theta1)
    mse = sum(squared_errors) / len(squared_errors)
    _ensure_finite(mse, "MSE")
    return mse


def main(argv: Sequence[str] | None = None) -> int:
    """Load a dataset and model, then print MAE and MSE."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        dataset = load_dataset(arguments.data)
        theta0, theta1 = load_model(arguments.model)
        mae = calculate_mae(dataset.mileage, dataset.prices, theta0, theta1)
        mse = calculate_mse(dataset.mileage, dataset.prices, theta0, theta1)
    except (DatasetError, ModelError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    return 0


def _calculate_errors(
    mileages: Sequence[float],
    prices: Sequence[float],
    theta0: float,
    theta1: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not mileages:
        raise ValueError("Precision requires at least one observation")
    if len(mileages) != len(prices):
        raise ValueError("Mileage and price columns must have the same length")
    if not all(
        math.isfinite(value) for value in (*mileages, *prices, theta0, theta1)
    ):
        raise ValueError("Precision values must all be finite")

    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    for mileage, price in zip(mileages, prices):
        error = estimate_price(theta0, theta1, mileage) - price
        absolute_error = abs(error)
        squared_error = error**2
        _ensure_finite(absolute_error, "absolute error")
        _ensure_finite(squared_error, "squared error")
        absolute_errors.append(absolute_error)
        squared_errors.append(squared_error)
    return tuple(absolute_errors), tuple(squared_errors)


def _ensure_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the prediction error of a linear regression model."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"CSV dataset path (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model JSON path (default: {DEFAULT_MODEL})",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
