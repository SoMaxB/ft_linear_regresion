import contextlib
import io
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


if __name__ == "__main__":
    unittest.main()
