# Training CLI

Issue #8 connects the existing data, training, and persistence components in
`train.py`. The command is intentionally an orchestrator; it does not contain
the regression formulas or JSON serialization logic.

## Usage

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
