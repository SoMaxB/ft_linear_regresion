# Training Loop

Issue #6 implements the training loop without coupling it to CSV files,
command-line arguments, or model persistence.

## API

```python
from src.regression import train_model

result = train_model(
    mileages=(100000.0, 120000.0, 140000.0),
    prices=(5000.0, 4600.0, 4200.0),
    learning_rate=0.1,
    iterations=1000,
)
```

`TrainingResult.theta0` and `TrainingResult.theta1` are ready for the public
hypothesis on original mileage values. The result also exposes:

- `normalized_theta0` and `normalized_theta1`, for mathematical diagnostics.
- `mean_km` and `std_km`, used by the internal transformation.
- `cost_history`, whose first value is the cost with both parameters at zero.

## Numerical method

Mileage is standardized with the population standard deviation:

```text
normalized_km = (km - mean_km) / std_km
```

Each iteration applies the subject's simultaneous update to normalized
mileage. Both gradients use the old parameters before either parameter is
changed. The cost is:

```text
J(theta) = 1 / (2 * m) * sum((theta0 + theta1 * normalized_km - price)^2)
```

After the final iteration, parameters are converted to the original scale:

```text
theta1_original = theta1_normalized / std_km
theta0_original = theta0_normalized - theta1_normalized * mean_km / std_km
```

This makes the following two predictions equivalent:

```text
theta0_original + theta1_original * km
theta0_normalized + theta1_normalized * normalized_km
```

## Rejected inputs

Training rejects empty or mismatched columns, non-finite observations, a
non-positive or non-finite learning rate, non-positive iteration counts, and
zero mileage standard deviation. Non-finite parameters or cost values are
also rejected immediately instead of being persisted or returned.

## Verification

`tests/test_regression.py` verifies that:

- training starts from zero and reduces the cost on a representative dataset;
- the hypothesis works for a single point and two points recover a known line;
- constant prices produce a zero slope and large mileage values remain finite;
- normalized and original-scale hypotheses produce the same predictions;
- empty, single-point, mismatched, non-finite, and zero-variance data are
  rejected;
- invalid and numerically unstable learning rates and iteration counts are
  rejected;
- returned parameters and the complete cost history remain finite.
