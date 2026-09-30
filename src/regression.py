"""Mathematical operations for the linear regression model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math


def estimate_price(theta0: float, theta1: float, mileage: float) -> float:
    """Estimate a car price with the subject's linear hypothesis.

    The formula is the same hypothesis required by the subject::

        estimatePrice(km) = theta0 + theta1 * km

    Training and input validation are intentionally handled elsewhere.
    """
    return theta0 + theta1 * mileage


def gradient_descent_step(
    theta0: float,
    theta1: float,
    learning_rate: float,
    mileages: Sequence[float],
    prices: Sequence[float],
) -> tuple[float, float]:
    """Return parameters after one simultaneous gradient descent step.

    For each observation, the subject defines the temporary gradients as::

        tmpTheta0 = learningRate * (1 / m) * sum(estimate - price)
        tmpTheta1 = learningRate * (1 / m) * sum((estimate - price) * km)

    Both gradients are calculated with the original parameters, then
    subtracted from them simultaneously. The feature values are used as
    provided; normalization belongs to the training loop.
    """
    if not mileages:
        raise ValueError("Gradient descent requires at least one observation")
    if len(mileages) != len(prices):
        raise ValueError("Mileage and price columns must have the same length")

    theta0_gradient = 0.0
    theta1_gradient = 0.0
    for mileage, price in zip(mileages, prices):
        error = estimate_price(theta0, theta1, mileage) - price
        theta0_gradient += error
        theta1_gradient += error * mileage

    scale = learning_rate / len(mileages)
    tmp_theta0 = scale * theta0_gradient
    tmp_theta1 = scale * theta1_gradient
    return theta0 - tmp_theta0, theta1 - tmp_theta1


@dataclass(frozen=True)
class TrainingResult:
    """Parameters and diagnostics produced by :func:`train_model`.

    ``theta0`` and ``theta1`` are always expressed in the original mileage
    scale. The normalized parameters are kept so tests and diagnostics can
    verify the conversion without changing the public prediction contract.
    """

    theta0: float
    theta1: float
    normalized_theta0: float
    normalized_theta1: float
    mean_km: float
    std_km: float
    cost_history: tuple[float, ...]


def train_model(
    mileages: Sequence[float],
    prices: Sequence[float],
    learning_rate: float = 0.1,
    iterations: int = 1000,
) -> TrainingResult:
    """Train a model with gradient descent on standardized mileage.

    Mileage is transformed with the population standard deviation before the
    subject gradient is applied::

        normalized_km = (km - mean_km) / std_km

    The returned ``theta0`` and ``theta1`` are converted back so they satisfy
    the public hypothesis on the original ``km`` values. The cost uses
    ``1 / (2 * m) * sum((estimate - price) ** 2)`` and includes its initial
    value before the first update.
    """
    _validate_training_inputs(mileages, prices, learning_rate, iterations)

    mean_km = sum(mileages) / len(mileages)
    variance = sum((mileage - mean_km) ** 2 for mileage in mileages) / len(
        mileages
    )
    std_km = math.sqrt(variance)
    if not math.isfinite(std_km) or std_km == 0.0:
        raise ValueError("Mileage standard deviation must be finite and non-zero")

    normalized_mileages = tuple(
        (mileage - mean_km) / std_km for mileage in mileages
    )
    theta0 = 0.0
    theta1 = 0.0
    cost_history = [
        _cost(theta0, theta1, normalized_mileages, prices)
    ]

    for _ in range(iterations):
        theta0, theta1 = gradient_descent_step(
            theta0,
            theta1,
            learning_rate,
            normalized_mileages,
            prices,
        )
        if not math.isfinite(theta0) or not math.isfinite(theta1):
            raise ValueError("Training produced a non-finite parameter")
        cost_history.append(_cost(theta0, theta1, normalized_mileages, prices))

    original_theta1 = theta1 / std_km
    original_theta0 = theta0 - theta1 * mean_km / std_km
    if not all(
        math.isfinite(value)
        for value in (original_theta0, original_theta1)
    ):
        raise ValueError("Training produced a non-finite original parameter")

    return TrainingResult(
        theta0=original_theta0,
        theta1=original_theta1,
        normalized_theta0=theta0,
        normalized_theta1=theta1,
        mean_km=mean_km,
        std_km=std_km,
        cost_history=tuple(cost_history),
    )


def _cost(
    theta0: float,
    theta1: float,
    mileages: Sequence[float],
    prices: Sequence[float],
) -> float:
    squared_error = sum(
        (estimate_price(theta0, theta1, mileage) - price) ** 2
        for mileage, price in zip(mileages, prices)
    )
    cost = squared_error / (2.0 * len(mileages))
    if not math.isfinite(cost):
        raise ValueError("Training produced a non-finite cost")
    return cost


def _validate_training_inputs(
    mileages: Sequence[float],
    prices: Sequence[float],
    learning_rate: float,
    iterations: int,
) -> None:
    if not mileages:
        raise ValueError("Training requires at least one observation")
    if len(mileages) != len(prices):
        raise ValueError("Mileage and price columns must have the same length")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("Learning rate must be finite and greater than zero")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("Iterations must be a positive integer")
    if iterations <= 0:
        raise ValueError("Iterations must be a positive integer")
    if not all(math.isfinite(value) for value in (*mileages, *prices)):
        raise ValueError("Training values must all be finite")
