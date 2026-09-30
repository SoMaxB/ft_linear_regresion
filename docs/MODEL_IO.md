# Model Persistence

Issue #7 implements the model-file contract without coupling it to training or
the command-line programs.

## File contract

The default path is `model/theta.json` and its complete JSON object is:

```json
{
  "theta0": 8490.12,
  "theta1": -0.0215
}
```

Both parameters are stored on the original mileage scale. The loader requires
exactly these two fields and validates that both values are finite numbers.
Extra fields, missing fields, strings, booleans, `NaN`, and infinities are
rejected with `ModelError`.

## API

```python
from src.model_io import load_model, save_model

save_model(8490.12, -0.0215)
theta0, theta1 = load_model()
```

`save_model` creates missing parent directories. It writes a temporary file in
the destination directory and atomically replaces the target, preventing an
interrupted save from leaving a partial JSON file.

`load_model` raises `ModelError` for a missing file, unreadable file, invalid
JSON, invalid schema, or non-finite values. A missing model is therefore an
explicit error rather than an implicit zero-parameter model.
