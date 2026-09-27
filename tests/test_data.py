import tempfile
import unittest
from pathlib import Path

from src.data import DatasetError, load_dataset


class DatasetLoadingTest(unittest.TestCase):
    def test_loads_project_dataset(self):
        dataset = load_dataset("data.csv")

        self.assertEqual(len(dataset), 24)
        self.assertEqual(dataset.mileage[0], 240000.0)
        self.assertEqual(dataset.prices[0], 3650.0)

    def test_accepts_whitespace_and_blank_lines(self):
        dataset = self._load_text("km,price\n 10 , 20 \n\n")

        self.assertEqual(dataset.mileage, (10.0,))
        self.assertEqual(dataset.prices, (20.0,))

    def test_rejects_invalid_header(self):
        self._assert_dataset_error("mileage,price\n10,20\n", "Invalid header")

    def test_rejects_empty_dataset(self):
        self._assert_dataset_error("km,price\n", "at least one")

    def test_rejects_incomplete_row(self):
        self._assert_dataset_error("km,price\n10\n", "expected 2 columns")

    def test_rejects_non_numeric_value(self):
        self._assert_dataset_error("km,price\nten,20\n", "expected a number")

    def test_rejects_non_finite_value(self):
        self._assert_dataset_error("km,price\n10,nan\n", "must be finite")

    def test_rejects_missing_file(self):
        with self.assertRaisesRegex(DatasetError, "Dataset not found"):
            load_dataset("does-not-exist.csv")

    def _load_text(self, content: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text(content, encoding="utf-8")
            return load_dataset(path)

    def _assert_dataset_error(self, content: str, message: str):
        with self.assertRaisesRegex(DatasetError, message):
            self._load_text(content)


if __name__ == "__main__":
    unittest.main()
