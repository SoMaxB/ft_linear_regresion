# Command-Line Applications

The Makefile provides shortcuts for the commands below. Run `make help` to
see the available targets and configurable variables.

Issue #8 connects the existing data, training, and persistence components in
`train.py`. Issue #9 adds the interactive prediction flow in `predict.py`.
Both commands are intentionally orchestrators; they do not contain the
regression formulas or JSON serialization logic.

## Training

From the repository root:

```bash
python3 train.py
```

This reads `data.csv`, trains with the standard training defaults, and writes
the original-scale parameters to `model/theta.json`.

Paths can be overridden for local experiments or tests:

```bash
python3 train.py --data path/to/data.csv --model path/to/theta.json
```

The default command remains small and deterministic. Optional controls can be
used when experimenting with the training behavior:

```bash
python3 train.py --learning-rate 0.05 --iterations 2500
```

The defaults remain `--learning-rate 0.1` and `--iterations 1000`.

The equivalent Make target is:

```bash
make train
```

For custom values:

```bash
make train DATA=path/to/data.csv MODEL=path/to/theta.json \
  LEARNING_RATE=0.05 ITERATIONS=2500
```

## Errors

Dataset, training, and model-file errors are printed as one concise `Error:`
message on standard error, and the command exits with status `1`. Successful
training prints the destination model path and exits with status `0`.

Argument parsing remains handled by `argparse`, including its standard usage
message for invalid command-line options.

## Prediction

From the repository root, after a model has been trained:

```bash
python3 predict.py
```

The command loads `model/theta.json`, asks for one mileage, and evaluates the
public hypothesis using the original-scale parameters:

```text
estimatePrice(km) = theta0 + theta1 * km
```

The model path can be overridden:

```bash
python3 predict.py --model path/to/theta.json
```

The equivalent shortcut is `make predict MODEL=path/to/theta.json`.

Successful output has the form `Estimated price: VALUE` and exits with status
`0`. The mileage must be numeric and finite; invalid input prints an `Error:`
message to standard error and exits with status `1`. A missing, unreadable, or
invalid model is also an explicit error and never falls back to zero
parameters.

## Bonus dataset plot

The first Bonus tool plots the `km,price` observations without adding an
external dependency. It generates SVG directly with the Python standard
library, which keeps it usable on clusters where installing packages is not
available:

```bash
python3 bonus/plot.py
```

The default input is `data.csv` and the default output is
`bonus/output/dataset.svg`. Both paths can be overridden:

```bash
python3 bonus/plot.py --data path/to/data.csv --output path/to/dataset.svg
```

## Bonus regression line

The same tool can load the persisted Mandatory model and draw its regression
line over the observations:

```bash
python3 bonus/plot.py --model model/theta.json
```

The equivalent shortcut is `make plot`. It expects a model at the default
path, so run `make train` first or provide `MODEL=...`.

The `--model` option is optional. Without it, the tool only draws the dataset,
which preserves the behavior of issue #13. When provided, the model is loaded
through `src.model_io.load_model`, so missing, malformed, or non-finite models
produce an explicit error instead of a misleading graph.

Training may standardize mileage internally, but `train_model` converts the
parameters back to the original scale before persistence. The plot therefore
uses the same public hypothesis as prediction:

```text
estimatePrice(km) = theta0 + theta1 * km
```

The SVG y-axis includes both observed prices and the line's endpoint
predictions, ensuring that the regression line remains visible even when it
falls outside the observed price range.

## Bonus model precision

The precision tool evaluates the persisted model against the same dataset:

```bash
python3 bonus/precision.py
```

Custom paths are supported:

```bash
python3 bonus/precision.py \
  --data path/to/data.csv \
  --model path/to/theta.json
```

The equivalent shortcut is `make precision MODEL=path/to/theta.json`.

It prints two regression error metrics:

- **MAE** (Mean Absolute Error) is the average absolute difference between the
  predicted and actual price. It uses the same units as `price`.
- **MSE** (Mean Squared Error) is the average squared difference. It uses
  squared price units and penalizes large errors more strongly.

Both metrics are `0` for perfect predictions. Lower values indicate a better
fit. They are error measurements, not classification accuracy percentages.
The tool uses the persisted original-scale parameters and the public
`estimatePrice(km)` hypothesis. It does not retrain or modify the model.

## Bonus training diagnostics

Training can optionally print cost checkpoints and export its diagnostics:

```bash
python3 train.py \
  --learning-rate 0.05 \
  --iterations 2500 \
  --progress \
  --diagnostics bonus/output/training.json
```

`--progress` prints evenly spaced checkpoints from the returned cost history.
It does not change the gradient descent algorithm. `--diagnostics` writes a
separate JSON document containing the configuration, original-scale and
normalized parameters, mileage normalization metadata, and complete cost
history. The Mandatory `model/theta.json` contract remains unchanged.

The equivalent shortcut is:

```bash
make diagnostics
```

Its output path can be overridden with `DIAGNOSTICS=path/to/training.json`.
