"""Mathematical operations for the linear regression model."""

from __future__ import annotations


def estimate_price(theta0: float, theta1: float, mileage: float) -> float:
    """Estimate a car price with the subject's linear hypothesis.

    The formula is the same hypothesis required by the subject::

        estimatePrice(km) = theta0 + theta1 * km

    Training and input validation are intentionally handled elsewhere.
    """
    return theta0 + theta1 * mileage
