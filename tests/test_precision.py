import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from bonus.precision import calculate_mae, calculate_mse, main
from src.model_io import save_model


class ModelPrecisionTest(unittest.TestCase):
    MILEAGES = (100.0, 200.0, 300.0)
    PRICES = (1000.0, 800.0, 700.0)

    def test_perfect_predictions_have_zero_error(self):
        theta0 = 1200.0
        theta1 = -1.0

        self.assertEqual(
            calculate_mae(self.MILEAGES, (1100.0, 1000.0, 900.0), theta0, theta1),
            0.0,
        )
        self.assertEqual(
            calculate_mse(self.MILEAGES, (1100.0, 1000.0, 900.0), theta0, theta1),
            0.0,
        )

    def test_metrics_match_manual_calculation(self):
        # Predictions are 1100, 1000, and 900; errors are 100, 200, and 200.
        theta0 = 1200.0
        theta1 = -1.0

        self.assertAlmostEqual(
            calculate_mae(self.MILEAGES, self.PRICES, theta0, theta1),
            500.0 / 3.0,
        )
        self.assertAlmostEqual(
            calculate_mse(self.MILEAGES, self.PRICES, theta0, theta1),
            90000.0 / 3.0,
        )

    def test_rejects_empty_observations(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            calculate_mae((), (), 0.0, 0.0)

    def test_rejects_mismatched_columns(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            calculate_mse((1.0,), (1.0, 2.0), 0.0, 0.0)

    def test_cli_loads_persisted_model_and_prints_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            model_path = root / "model" / "theta.json"
            data_path.write_text(
                "km,price\n100,1000\n200,800\n300,700\n",
                encoding="utf-8",
            )
            save_model(1200.0, -1.0, model_path)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = main(
                    ["--data", str(data_path), "--model", str(model_path)]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Mean Absolute Error (MAE): 166.666667", output.getvalue())
            self.assertIn("Mean Squared Error (MSE): 30000.000000", output.getvalue())

    def test_cli_reports_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            data_path.write_text("km,price\n100,1000\n", encoding="utf-8")
            error_output = io.StringIO()

            with contextlib.redirect_stderr(error_output):
                exit_code = main(
                    ["--data", str(data_path), "--model", str(root / "missing.json")]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("Model not found", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
