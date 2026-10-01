# Mandatory Audit

This audit compares the implementation with the Mandatory requirements in
`en.subject.pdf`. It is intentionally completed before any Bonus work.

## Checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| Both required programs exist | PASS | `train.py` trains and saves the model; `predict.py` prompts for mileage and prints a price. |
| Required hypothesis is used | PASS | `src.regression.estimate_price` evaluates `theta0 + theta1 * mileage`; prediction calls this function. |
| Required training function is used | PASS | `src.regression.gradient_descent_step` calculates the two subject gradients and `train_model` applies them repeatedly. |
| Parameters update simultaneously | PASS | Both gradients are accumulated from the old parameters before either new parameter is returned. |
| No complete-regression library is used | PASS | Mandatory code uses the Python standard library; no `numpy.polyfit`, `sklearn.LinearRegression`, or equivalent is present. |
| Complete flow works | PASS | `tests/test_integration.py` verifies dataset loading, training, persistence, and multiple predictions in temporary paths. |
| Names and commands are documented | PASS | `README.md`, `docs/CLI.md`, `docs/TESTING.md`, and `Makefile` document entry points and verification commands. |
| Mathematical decisions are explainable | PASS | `docs/DECISIONS.md` and `docs/TRAINING.md` explain the formulas, normalization, conversion, validation, and numerical guards. |

## Subject Formula Mapping

The subject's temporary values are implemented in
`gradient_descent_step`:

```text
tmpTheta0 = learningRate * (1 / m) * sum(estimatePrice - price)
tmpTheta1 = learningRate * (1 / m) * sum((estimatePrice - price) * mileage)
```

The function returns both updates at once:

```text
theta0 - tmpTheta0, theta1 - tmpTheta1
```

The public prediction remains exactly:

```text
estimatePrice(mileage) = theta0 + theta1 * mileage
```

## Internal Standardization

The training loop standardizes mileage before applying the subject gradient to
avoid unstable updates with mileage values around hundreds of thousands. After
training it converts the parameters back:

```text
theta1_original = theta1_normalized / std_km
theta0_original = theta0_normalized - theta1_normalized * mean_km / std_km
```

The persisted model therefore belongs to the original mileage scale and the
prediction program does not need normalization metadata. Tests verify that the
normalized and original-scale hypotheses produce the same predictions.

## Verification

The final local verification commands are:

```bash
make check
make test
```

They pass with 62 tests. The test suite covers the mathematical core, invalid
inputs, model persistence, both applications, and the complete train-to-predict
flow. Bonus work must remain blocked until this audit is accepted.
