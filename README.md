# ft_linear_regression

Implementation of the 42 `ft_linear_regression` project.

The project trains a simple linear regression with gradient descent to predict
the price of a car from its mileage.

## Status

The mathematical core currently includes the linear hypothesis, one gradient
descent step, the standardized training loop, and JSON model persistence. The
command-line train/predict flow is planned in the following issues.

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
- `docs/MODEL_IO.md`: model JSON contract and persistence behavior.
- `src/model_io.py`: model JSON persistence and validation.
- `docs/ISSUE_PLAN.md`: implementation roadmap and acceptance criteria.

## Development commands

```bash
make check
make test
```

## Constraints

The regression algorithm must be implemented by us. Libraries that perform
the complete regression are not allowed. Bonus visualization is intentionally
kept separate from the Mandatory implementation.
