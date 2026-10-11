# ft_linear_regression

Implementation of the 42 `ft_linear_regression` project.

The project trains a simple linear regression with gradient descent to predict
the price of a car from its mileage.

## Status

The Mandatory implementation currently includes the linear hypothesis, one
gradient descent step, the standardized training loop, JSON model persistence,
and both command-line flows for training and prediction.

## Interface

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
- `docs/AUDIT.md`: Mandatory subject compliance audit.
- `mandatory_evaluation.md`: terminal command script for the peer evaluation.
- `bonus_evaluation.md`: Bonus commands for visualization, precision, and diagnostics.
- `docs/evaluation.html`: interactive single-page evaluation guide.
- `docs/MODEL_IO.md`: model JSON contract and persistence behavior.
- `docs/CLI.md`: command-line training behavior and error handling.
- `src/model_io.py`: model JSON persistence and validation.
- `docs/ISSUE_PLAN.md`: implementation roadmap and acceptance criteria.

## Development commands

```bash
make help    # Show all available commands
make verify  # Compile and run the complete test suite
```

The Makefile also provides shortcuts for the complete workflow:

```bash
make train
make predict
make plot
make precision
make diagnostics
```

Common values can be overridden without editing files:

```bash
make train DATA=data.csv MODEL=/tmp/theta.json \
  LEARNING_RATE=0.05 ITERATIONS=2500
```

Run `make help` for the complete command list. Bonus evaluation steps are in
`bonus_evaluation.md`; the Mandatory walkthrough is in
`mandatory_evaluation.md`.

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

## Bonus visualization

The optional dataset plot uses only the Python standard library and writes an
SVG file, so it works on the school clusters without `sudo`, `apt`, or `pip`:

```bash
python3 bonus/plot.py
```

The default output is `bonus/output/dataset.svg`. Open that file in a browser or
any SVG-capable image viewer. Custom paths can be supplied with `--data` and
`--output`.

Add a persisted Mandatory model to draw its regression line on the same plot:

```bash
python3 bonus/plot.py --model model/theta.json
```

The model parameters are already stored on the original mileage scale, so the
plot uses the public hypothesis directly and does not need normalization data.

Measure model error with the optional precision tool:

```bash
python3 bonus/precision.py
```

It reports MAE, the average absolute error in price units, and MSE, the average
squared error in squared price units. Lower values are better; neither metric
is a percentage of correct predictions.

Training controls and diagnostics are optional:

```bash
python3 train.py --learning-rate 0.05 --iterations 2500 --progress \
  --diagnostics bonus/output/training.json
```

Without these options, the Mandatory training command and model format are
unchanged.

## Constraints

The regression algorithm must be implemented by us. Libraries that perform
the complete regression are not allowed. Bonus visualization is intentionally
kept separate from the Mandatory implementation.
