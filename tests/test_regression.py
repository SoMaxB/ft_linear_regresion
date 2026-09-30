import unittest

from src.regression import estimate_price, gradient_descent_step


class LinearHypothesisTest(unittest.TestCase):
    def test_zero_parameters_produce_zero_price(self):
        self.assertEqual(estimate_price(0.0, 0.0, 240000.0), 0.0)

    def test_evaluates_known_parameters(self):
        self.assertEqual(estimate_price(1000.0, -0.5, 1200.0), 400.0)

    def test_supports_decimal_values(self):
        result = estimate_price(8490.12, -0.0215, 100000.0)

        self.assertAlmostEqual(result, 6340.12)

    def test_gradient_step_matches_manual_calculation(self):
        new_theta0, new_theta1 = gradient_descent_step(
            theta0=0.0,
            theta1=0.0,
            learning_rate=0.1,
            mileages=(0.0, 10.0),
            prices=(10.0, 20.0),
        )

        self.assertEqual(new_theta0, 1.5)
        self.assertEqual(new_theta1, 10.0)

    def test_zero_learning_rate_keeps_parameters(self):
        result = gradient_descent_step(
            theta0=3.5,
            theta1=-0.25,
            learning_rate=0.0,
            mileages=(100.0, 200.0),
            prices=(20.0, 30.0),
        )

        self.assertEqual(result, (3.5, -0.25))

    def test_gradient_step_rejects_empty_observations(self):
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            gradient_descent_step(0.0, 0.0, 0.1, (), ())

    def test_gradient_step_rejects_mismatched_columns(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            gradient_descent_step(0.0, 0.0, 0.1, (10.0,), (20.0, 30.0))


if __name__ == "__main__":
    unittest.main()
