# Architecture

## Data flow

```text
data.csv
   |
   v
src/data.py ----> src/regression.py ----> src/model_io.py
                                              |
                                              v
                                      model/theta.json
                                              |
                                              v
                                         predict.py
```

## Project tree

```text
.
├── data.csv
├── train.py
├── predict.py
├── model/
│   └── theta.json              # generated, ignored by Git
├── src/
│   ├── data.py
│   ├── model_io.py
│   ├── regression.py
│   ├── validation.py
│   └── diagnostics.py
├── tests/
│   ├── test_data.py
│   ├── test_predict.py
│   └── test_regression.py
└── bonus/
    ├── plot.py
    └── precision.py
```

## Responsibilities

### `train.py`

Orchestrates loading the dataset, training the model, and persisting the
original-scale parameters. It should not contain the gradient formulas.

### `predict.py`

Loads the model, validates one mileage input, and evaluates the public
hypothesis. It should not contain training logic.

### `src/data.py`

Owns CSV parsing, header validation, numeric conversion, and dataset errors.

### `src/regression.py`

Owns the hypothesis, normalization conversion, gradient calculation,
simultaneous parameter updates, and the pure training loop. `train_model`
returns original-scale parameters plus the normalization metadata and cost
history needed by the caller or by diagnostics.

### `src/model_io.py`

Owns JSON serialization, model loading, model-file errors, and creation of the
model directory. It writes through a temporary file and atomic replacement so
an interrupted save does not leave a partial JSON model.

### `src/validation.py`

Owns reusable checks for finite numeric values and user input.

### `src/diagnostics.py`

Owns optional training-diagnostics JSON serialization. It does not participate
in model loading or prediction, and writes diagnostics atomically to a separate
path.

### `tests/`

Tests the mathematical core independently from command-line input and file
system details, then covers the complete train-to-predict flow.

### `bonus/`

Contains optional visualization and precision tools. It must not become a
dependency of Mandatory programs.

## Error policy

Errors should be explicit and actionable. The programs must reject malformed
CSV rows, non-numeric input, non-finite values, empty datasets, unreadable
models, and datasets whose mileage has zero variance.

## Training contract

`train_model` starts both normalized parameters at zero. It computes the
population mean and standard deviation of mileage, applies gradient descent to
the normalized values, and records the cost before the first update and after
each update. The cost is:

```text
J(theta) = 1 / (2 * m) * sum((estimatePrice(normalized_km) - price)^2)
```

The function does not read files or write the model. Those responsibilities
remain in the application and persistence issues. A non-positive learning
rate, invalid iteration count, zero mileage variance, or any non-finite
intermediate/result is rejected with `ValueError`.
