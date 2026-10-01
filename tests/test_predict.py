import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import predict
from src.model_io import save_model


class PredictionApplicationTest(unittest.TestCase):
    def test_main_loads_model_and_evaluates_original_scale_hypothesis(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "theta.json"
            save_model(10000.0, -0.02, model_path)
            output = io.StringIO()

            with patch("builtins.input", return_value="250000"), contextlib.redirect_stdout(output):
                exit_code = predict.main(["--model", str(model_path)])

            self.assertEqual(exit_code, 0)
            self.assertIn("Estimated price: 5000.0", output.getvalue())

    def test_main_rejects_non_numeric_and_non_finite_mileage(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "theta.json"
            save_model(10000.0, -0.02, model_path)

            invalid_values = (
                ("not-a-number", "must be a number"),
                ("nan", "must be finite"),
            )
            for value, message in invalid_values:
                with self.subTest(value=value):
                    error_output = io.StringIO()
                    with (
                        patch("builtins.input", return_value=value),
                        contextlib.redirect_stderr(error_output),
                    ):
                        exit_code = predict.main(["--model", str(model_path)])

                    self.assertEqual(exit_code, 1)
                    self.assertIn(f"Mileage {message}", error_output.getvalue())

    def test_main_reports_missing_model_without_traceback(self):
        error_output = io.StringIO()

        with contextlib.redirect_stderr(error_output):
            exit_code = predict.main(["--model", "missing.json"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Model not found", error_output.getvalue())
        self.assertNotIn("Traceback", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
