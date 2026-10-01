# Technical Decisions

## D-001: Python

Python is the project language. It keeps the mathematical implementation
readable and makes the optional visualization straightforward.

## D-002: Dataset contract

The default dataset is `data.csv` with this schema:

```csv
km,price
240000,3650
139800,3650
```

The first row is a required header. Each following row contains two finite
numeric values: mileage in `km` and target price in `price`.

The subject does not mandate the filename or header names. This repository
uses this convention so that execution, tests, and peer evaluation are
deterministic.

## D-003: Model contract

The trained model is stored at `model/theta.json`:

```json
{
  "theta0": 8490.12,
  "theta1": -0.0215
}
```

The persisted parameters always belong to the original mileage scale. The
prediction program therefore does not need to know how training was
internally normalized.

The persistence layer requires exactly the `theta0` and `theta1` fields and
rejects non-numeric or non-finite values. Saving creates missing parent
directories and replaces the destination atomically.

## D-004: Public hypothesis

Both programs use the subject hypothesis:

```text
estimatePrice(km) = theta0 + theta1 * km
```

The initial parameters are `theta0 = 0` and `theta1 = 0`.

## D-005: Internal mileage standardization

Training will standardize mileage when this is needed for numerical stability:

```text
x_normalized = (km - mean_km) / std_km
```

Gradient descent will operate on the normalized feature. Once training ends,
the normalized parameters will be converted to equivalent parameters on the
original scale:

```text
theta1_original = theta1_normalized / std_km
theta0_original = theta0_normalized - theta1_normalized * mean_km / std_km
```

This preserves the exact public hypothesis while preventing the large mileage
values from making the gradient unstable. A zero standard deviation is an
invalid training dataset because all mileage values are identical.

The implementation must include tests proving equivalence between the
normalized model and the persisted original-scale model.

## D-006: Dependency policy

Mandatory code will use the Python standard library where practical. A
visualization library may be introduced only for the Bonus. No library may
perform the complete regression or hide the specified gradient descent.

## D-007: Mandatory before Bonus

Bonus work starts only after the complete Mandatory flow passes mathematical,
error-handling, integration, and subject-compliance checks.

## D-008: Interactive prediction errors

`predict.py` prompts for one mileage and evaluates the original-scale model.
It rejects non-numeric and non-finite input, reports model-file failures on
standard error, and exits with status `1`. It never substitutes zero parameters
when `model/theta.json` is missing or invalid.

## D-009: Bonus error metrics

The Bonus precision tool reports Mean Absolute Error (MAE) and Mean Squared
Error (MSE), rather than presenting regression quality as a percentage. MAE is
the primary human-readable metric because it remains in the same units as the
price. MSE is included because it penalizes large errors more strongly.

```text
MAE = 1 / m * sum(abs(estimatePrice(km) - price))
MSE = 1 / m * sum((estimatePrice(km) - price)^2)
```

Both metrics use the persisted original-scale parameters and do not alter the
Mandatory training algorithm. Lower values are better and zero is perfect.
