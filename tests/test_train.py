import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import train
from src.model_io import load_model


class TrainingApplicationTest(unittest.TestCase):
    DATA = "km,price\n100000,5000\n120000,4600\n140000,4200\n"

    def test_main_trains_and_saves_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            model_path = root / "output" / "theta.json"
            data_path.write_text(self.DATA, encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = train.main(
                    ["--data", str(data_path), "--model", str(model_path)]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Model trained and saved", output.getvalue())
            theta0, theta1 = load_model(model_path)
            self.assertTrue(theta0 > 0.0)
            self.assertTrue(theta1 < 0.0)

    def test_main_reports_dataset_errors_without_traceback(self):
        error_output = io.StringIO()

        with contextlib.redirect_stderr(error_output):
            exit_code = train.main(["--data", "missing-data.csv"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Dataset not found", error_output.getvalue())
        self.assertNotIn("Traceback", error_output.getvalue())

    def test_optional_training_controls_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            model_path = root / "theta.json"
            diagnostics_path = root / "diagnostics" / "training.json"
            data_path.write_text(self.DATA, encoding="utf-8")
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = train.main(
                    [
                        "--data",
                        str(data_path),
                        "--model",
                        str(model_path),
                        "--learning-rate",
                        "0.05",
                        "--iterations",
                        "20",
                        "--progress",
                        "--diagnostics",
                        str(diagnostics_path),
                    ]
                )

            payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            output_lines = output.getvalue().splitlines()
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["learning_rate"], 0.05)
            self.assertEqual(payload["iterations"], 20)
            self.assertEqual(len(payload["cost_history"]), 21)
            self.assertTrue(any("Iteration 0/20" in line for line in output_lines))
            self.assertTrue(any("Iteration 20/20" in line for line in output_lines))
            self.assertIn("Model trained and saved", output.getvalue())

    def test_default_training_output_does_not_include_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            model_path = root / "theta.json"
            data_path.write_text(self.DATA, encoding="utf-8")
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = train.main(
                    ["--data", str(data_path), "--model", str(model_path)]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                output.getvalue(),
                f"Model trained and saved to {model_path}\n",
            )


if __name__ == "__main__":
    unittest.main()
