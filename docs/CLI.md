# Command-Line Applications

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

The CLI does not expose learning-rate or iteration controls yet. Optional
training diagnostics belong to a later issue so the Mandatory flow remains
small and deterministic.

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
