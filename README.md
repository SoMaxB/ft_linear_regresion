# ft_linear_regression

Implementation of the 42 `ft_linear_regression` project.

The project trains a simple linear regression with gradient descent to predict
the price of a car from its mileage.

## Status

The Mandatory implementation currently includes the linear hypothesis, one
gradient descent step, the standardized training loop, JSON model persistence,
and both command-line flows for training and prediction.

## Planned interface

```text
data.csv -> train.py -> model/theta.json -> predict.py
```

The public hypothesis is:

```text
estimatePrice(km) = theta0 + theta1 * km
```

## Documentation

- `en.subject.pdf`: authoritative project subject.
- `docs/DECISIONS.md`: agreed technical decisions and rationale.
- `docs/ARCHITECTURE.md`: planned structure and responsibilities.
- `docs/TRAINING.md`: training-loop formulas, API, and review notes.
- `docs/TESTING.md`: unit, application, and integration test coverage.
- `docs/MODEL_IO.md`: model JSON contract and persistence behavior.
- `docs/CLI.md`: command-line training behavior and error handling.
- `src/model_io.py`: model JSON persistence and validation.
- `docs/ISSUE_PLAN.md`: implementation roadmap and acceptance criteria.

## Development commands

```bash
make check
make test
```

Train from the repository root with the default paths:

```bash
python3 train.py
```

Use `--data PATH` and `--model PATH` to override them.

Predict from a trained model:

```bash
python3 predict.py
```

Use `--model PATH` to load a model from another location.

## Constraints

The regression algorithm must be implemented by us. Libraries that perform
the complete regression are not allowed. Bonus visualization is intentionally
kept separate from the Mandatory implementation.
