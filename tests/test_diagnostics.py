import json
import tempfile
import unittest
from pathlib import Path

from src.diagnostics import DiagnosticsError, save_training_diagnostics
from src.regression import train_model


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_saves_configuration_and_cost_history(self):
        result = train_model(
            (100000.0, 120000.0, 140000.0),
            (5000.0, 4600.0, 4200.0),
            learning_rate=0.05,
            iterations=20,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "diagnostics.json"
            save_training_diagnostics(result, 0.05, 20, path)

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["learning_rate"], 0.05)
        self.assertEqual(payload["iterations"], 20)
        self.assertEqual(len(payload["cost_history"]), 21)
        self.assertEqual(payload["initial_cost"], payload["cost_history"][0])
        self.assertEqual(payload["final_cost"], payload["cost_history"][-1])
        self.assertLess(payload["final_cost"], payload["initial_cost"])

    def test_rejects_cost_history_with_wrong_length(self):
        result = train_model(
            (100000.0, 120000.0, 140000.0),
            (5000.0, 4600.0, 4200.0),
            iterations=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DiagnosticsError, "length"):
                save_training_diagnostics(result, 0.1, 3, Path(directory) / "out.json")


if __name__ == "__main__":
    unittest.main()
