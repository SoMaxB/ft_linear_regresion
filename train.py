"""Command-line entry point for model training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from src.data import DatasetError, load_dataset
from src.diagnostics import DiagnosticsError, save_training_diagnostics
from src.model_io import ModelError, save_model
from src.regression import train_model


def main(argv: Sequence[str] | None = None) -> int:
    """Load data, train the model, and persist its original-scale parameters."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        dataset = load_dataset(arguments.data)
        result = train_model(
            dataset.mileage,
            dataset.prices,
            learning_rate=arguments.learning_rate,
            iterations=arguments.iterations,
        )
        save_model(result.theta0, result.theta1, arguments.model)
        if arguments.diagnostics is not None:
            save_training_diagnostics(
                result,
                arguments.learning_rate,
                arguments.iterations,
                arguments.diagnostics,
            )
    except (DatasetError, DiagnosticsError, ModelError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if arguments.progress:
        _print_progress(result.cost_history, arguments.iterations)
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Gradient descent learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of gradient descent iterations (default: 1000)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print cost checkpoints after training",
    )
    parser.add_argument(
        "--diagnostics",
        help="Optional JSON path for training diagnostics",
    )
    return parser


def _print_progress(cost_history: tuple[float, ...], iterations: int) -> None:
    """Print ten evenly spaced cost checkpoints without changing training."""
    step = max(1, iterations // 10)
    checkpoints = set(range(0, iterations + 1, step))
    checkpoints.add(iterations)
    for iteration in sorted(checkpoints):
        print(
            f"Iteration {iteration}/{iterations} - "
            f"cost: {cost_history[iteration]:.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
