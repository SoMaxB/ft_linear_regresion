# Issue Plan

The GitHub issues follow this order. Each implementation issue must satisfy
its acceptance criteria before the next dependent issue starts.

## Preparation

- `#1` Define project contract and decisions.
- `#2` Create the application skeleton and development commands.

## Mandatory: data and mathematics

- `#3` Implement CSV loading and validation.
- `#4` Implement the linear hypothesis.
- `#5` Implement the gradient descent step.
- `#6` Implement the training loop and numerical stability.
- `#7` Implement model persistence.

## Mandatory: applications and verification

- `#8` Implement `train.py`.
- `#9` Implement `predict.py` (implemented).
- `#10` Add mathematical unit tests (implemented).
- `#11` Add train-to-predict integration tests (implemented).
- `#12` Perform the Mandatory subject audit (implemented).

## Bonus

- `#13` Plot the dataset (implemented).
- `#14` Plot the regression line with the dataset (implemented).
- `#15` Calculate model precision (implemented).
- `#16` Improve optional CLI and training diagnostics (implemented).

Bonus issues are blocked by `#12`. The exact issue dependencies and acceptance
criteria are recorded in GitHub.
