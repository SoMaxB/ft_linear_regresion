# ft_linear_regression

Implementation of the 42 `ft_linear_regression` project.

The project trains a simple linear regression with gradient descent to predict
the price of a car from its mileage.

## Status

The repository is currently in the preparation phase. The subject, technical
decisions, architecture, and implementation issues are defined; application
code will be added issue by issue.

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
- `docs/ISSUE_PLAN.md`: implementation roadmap and acceptance criteria.

## Constraints

The regression algorithm must be implemented by us. Libraries that perform
the complete regression are not allowed. Bonus visualization is intentionally
kept separate from the Mandatory implementation.
