# Bonus Evaluation Commands

Run these commands from the repository root after the Mandatory evaluation has
passed. The Bonus tools use only the Python standard library, so no package
installation is required.

## 1. Show the available shortcuts

```bash
make help
```

The main workflow targets are:

- `make train`: train the model.
- `make plot`: generate the dataset and regression-line SVG.
- `make precision`: calculate MAE and MSE.
- `make diagnostics`: show cost checkpoints and export training diagnostics.

## 2. Train with the default configuration

```bash
make train
cat model/theta.json
```

The persisted model contains the original-scale `theta0` and `theta1`.

## 3. Generate the dataset and regression-line plot

```bash
make plot
```

The default output is:

```text
bonus/output/regression.svg
```

Open the SVG in a browser. Custom paths are supported:

```bash
make plot DATA=data.csv MODEL=model/theta.json \
  PLOT_OUTPUT=/tmp/regression.svg
```

## 4. Calculate model precision

```bash
make precision
```

The command reports:

- MAE in price units.
- MSE in squared price units.

Lower values are better and zero is a perfect fit.

## 5. Export training diagnostics

```bash
make diagnostics
```

This prints cost checkpoints and creates:

```text
bonus/output/training.json
```

Custom training values and output paths can be supplied:

```bash
make diagnostics LEARNING_RATE=0.05 ITERATIONS=2500 \
  DIAGNOSTICS=/tmp/training.json
```

The diagnostic JSON includes the configuration, original- and normalized-scale
parameters, normalization metadata, initial and final cost, and the complete
cost history.

## 6. Run all verification

```bash
make verify
```

This compiles the project and runs the complete unit and integration suite.

## 7. Verify the Mandatory/Bonus boundary

The following commands must continue to work without Bonus output files:

```bash
make train
printf '100000\n' | make predict
```

Bonus visualization and diagnostics are optional and do not change the
Mandatory model format or prediction hypothesis.
