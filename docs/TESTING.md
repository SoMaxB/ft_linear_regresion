# Testing

The project separates mathematical unit tests from application and integration
tests. All tests use temporary files for generated datasets and models, so they
do not depend on a previously trained `model/theta.json`.

## Commands

Run the complete verification suite from the repository root:

```bash
make check
make test
```

## Coverage

- `tests/test_regression.py` checks the hypothesis, gradient update, training
  convergence, normalization conversion, numerical stability, and invalid
  mathematical inputs.
- `tests/test_train.py` checks the training application and its error output.
- `tests/test_predict.py` checks model loading, input validation, hypothesis
  evaluation, and missing-model behavior.
- `tests/test_integration.py` runs `data.csv -> train.py -> theta.json ->
  predict.py` with several mileages and a controlled dataset.
- `tests/test_precision.py` checks MAE and MSE against manually calculated
  values, perfect predictions, invalid inputs, model loading, and CLI errors.
- `tests/test_diagnostics.py` checks atomic diagnostics serialization, cost
  history metadata, and invalid history lengths.

The integration tests also verify that prediction fails explicitly before a
model has been trained and does not request mileage input in that case.
