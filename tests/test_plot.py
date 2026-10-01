import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from bonus.plot import create_dataset_svg, main
from src.data import Dataset


class DatasetPlotTest(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset((100000.0, 150000.0, 200000.0), (8000.0, 7000.0, 6000.0))

    def test_creates_svg_with_points_axes_and_labels(self):
        svg = create_dataset_svg(self.dataset)

        self.assertIn('<svg xmlns="http://www.w3.org/2000/svg"', svg)
        self.assertEqual(svg.count("<circle"), 3)
        self.assertIn("Mileage (km)", svg)
        self.assertIn(">Price</text>", svg)
        self.assertIn("Mileage and price dataset", svg)

    def test_cli_loads_dataset_and_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            output_path = root / "nested" / "dataset.svg"
            data_path.write_text(
                "km,price\n100000,8000\n150000,7000\n200000,6000\n",
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    ["--data", str(data_path), "--output", str(output_path)]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            self.assertIn("Dataset plot written", output.getvalue())

    def test_cli_reports_invalid_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "invalid.csv"
            output_path = Path(directory) / "dataset.svg"
            data_path.write_text("wrong,header\n1,2\n", encoding="utf-8")
            error_output = io.StringIO()

            with contextlib.redirect_stderr(error_output):
                exit_code = main(
                    ["--data", str(data_path), "--output", str(output_path)]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("Invalid header", error_output.getvalue())
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
