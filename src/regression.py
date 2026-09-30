"""Mathematical operations for the linear regression model."""

from __future__ import annotations

from collections.abc import Sequence


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
