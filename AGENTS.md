# Repository Guidelines

## Project

This repository implements the 42 `ft_linear_regression` subject. The goal is
to train a one-feature linear regression with gradient descent and use it to
predict a car price from its mileage.

## Constraints

- The Mandatory implementation must use the subject hypothesis and formulas.
- Do not use libraries that perform the complete regression, such as
  `numpy.polyfit` or `sklearn.LinearRegression`.
- Keep Mandatory code independent from Bonus visualization and metrics.
- Every mathematical decision must be understandable during peer evaluation.

## Current decisions

- Language: Python.
- Dataset: `data.csv` with `km,price` columns.
- Entry points: `train.py` and `predict.py`.
- Persisted model: `model/theta.json`.
- Training may standardize mileage internally for numerical stability, then
  convert parameters back to the original hypothesis before persisting them.

See `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`, and `docs/ISSUE_PLAN.md`.
