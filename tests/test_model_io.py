import json
import tempfile
import unittest
from pathlib import Path

from src.model_io import ModelError, load_model, save_model


class ModelPersistenceTest(unittest.TestCase):
    def test_save_creates_directory_and_round_trips_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "theta.json"

            save_model(8490.12, -0.0215, path)

            self.assertEqual(load_model(path), (8490.12, -0.0215))
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"theta0": 8490.12, "theta1": -0.0215},
            )

    def test_missing_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"

            with self.assertRaisesRegex(ModelError, "Model not found"):
                load_model(path)

    def test_corrupt_json_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theta.json"
            path.write_text("{not-json", encoding="utf-8")

            with self.assertRaisesRegex(ModelError, "Invalid JSON"):
                load_model(path)

    def test_rejects_invalid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theta.json"
            path.write_text('{"theta0": 1}', encoding="utf-8")

            with self.assertRaisesRegex(ModelError, "exactly"):
                load_model(path)

    def test_rejects_non_finite_loaded_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theta.json"
            path.write_text('{"theta0": NaN, "theta1": 1}', encoding="utf-8")

            with self.assertRaisesRegex(ModelError, "finite"):
                load_model(path)

    def test_rejects_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theta.json"
            path.write_bytes(b"\xff\xfe")

            with self.assertRaisesRegex(ModelError, "UTF-8"):
                load_model(path)

    def test_rejects_non_finite_values_when_saving(self):
        with self.assertRaisesRegex(ModelError, "finite"):
            save_model(float("inf"), 0.0)


if __name__ == "__main__":
    unittest.main()
