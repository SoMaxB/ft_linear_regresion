import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import predict
import train


class TrainToPredictIntegrationTest(unittest.TestCase):
    DATA = "km,price\n100000,8000\n150000,7000\n200000,6000\n"

    def test_training_model_is_used_for_multiple_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            model_path = root / "model" / "theta.json"
            data_path.write_text(self.DATA, encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                train_exit_code = train.main(
                    ["--data", str(data_path), "--model", str(model_path)]
                )

            self.assertEqual(train_exit_code, 0)
            self.assertTrue(model_path.is_file())

            expected_prices = ((100000, 8000.0), (150000, 7000.0), (200000, 6000.0))
            for mileage, expected_price in expected_prices:
                with self.subTest(mileage=mileage):
                    output = io.StringIO()
                    with (
                        patch("builtins.input", return_value=str(mileage)),
                        contextlib.redirect_stdout(output),
                    ):
                        predict_exit_code = predict.main(
                            ["--model", str(model_path)]
                        )

                    self.assertEqual(predict_exit_code, 0)
                    prediction = float(output.getvalue().split(": ", 1)[1])
                    self.assertAlmostEqual(
                        prediction,
                        expected_price,
                        places=6,
                    )

    def test_prediction_fails_before_training_when_model_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model" / "theta.json"
            error_output = io.StringIO()

            with (
                patch("builtins.input") as input_mock,
                contextlib.redirect_stderr(error_output),
            ):
                exit_code = predict.main(["--model", str(model_path)])

            self.assertEqual(exit_code, 1)
            self.assertIn("Error: Model not found", error_output.getvalue())
            input_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
