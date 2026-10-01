"""Interactive command-line entry point for price prediction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
import sys

from src.model_io import ModelError, load_model
from src.regression import estimate_price
from src.validation import parse_finite_float


def main(argv: Sequence[str] | None = None) -> int:
    """Read one mileage, evaluate the public hypothesis, and print the price."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        theta0, theta1 = load_model(arguments.model)
        mileage = parse_finite_float(input("Enter a mileage: "), "Mileage")
        prediction = estimate_price(theta0, theta1, mileage)
        if not math.isfinite(prediction):
            raise ValueError("Prediction is not finite")
    except (EOFError, ModelError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Estimated price: {prediction}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a car price from its mileage."
    )
    parser.add_argument(
        "--model",
        default="model/theta.json",
        help="Input model path (default: model/theta.json)",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
