"""Command-line entry point for model training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from src.data import DatasetError, load_dataset
from src.model_io import ModelError, save_model
from src.regression import train_model


def main(argv: Sequence[str] | None = None) -> int:
    """Load data, train the model, and persist its original-scale parameters."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        dataset = load_dataset(arguments.data)
        result = train_model(dataset.mileage, dataset.prices)
        save_model(result.theta0, result.theta1, arguments.model)
    except (DatasetError, ModelError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Model trained and saved to {arguments.model}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the mileage-to-price linear regression model."
    )
    parser.add_argument(
        "--data",
        default="data.csv",
        help="CSV dataset path (default: data.csv)",
    )
    parser.add_argument(
        "--model",
        default="model/theta.json",
        help="Output model path (default: model/theta.json)",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
