import unittest

from src.regression import estimate_price


class LinearHypothesisTest(unittest.TestCase):
    def test_zero_parameters_produce_zero_price(self):
        self.assertEqual(estimate_price(0.0, 0.0, 240000.0), 0.0)

    def test_evaluates_known_parameters(self):
        self.assertEqual(estimate_price(1000.0, -0.5, 1200.0), 400.0)

    def test_supports_decimal_values(self):
        result = estimate_price(8490.12, -0.0215, 100000.0)

        self.assertAlmostEqual(result, 6340.12)


if __name__ == "__main__":
    unittest.main()
