import math
import unittest

from src.regression import estimate_price, gradient_descent_step, train_model


class LinearHypothesisTest(unittest.TestCase):
    def test_evaluates_a_single_point(self):
        self.assertEqual(estimate_price(10.0, 2.0, 5.0), 20.0)

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


class TrainingLoopTest(unittest.TestCase):
    MILEAGES = (100000.0, 120000.0, 140000.0, 160000.0)
    PRICES = (5000.0, 4600.0, 4200.0, 3800.0)

    def test_cost_decreases_from_initial_parameters(self):
        result = train_model(self.MILEAGES, self.PRICES, iterations=100)

        expected_initial_cost = sum(price**2 for price in self.PRICES) / (
            2 * len(self.PRICES)
        )
        self.assertEqual(result.cost_history[0], expected_initial_cost)
        self.assertLess(result.cost_history[-1], result.cost_history[0])
        self.assertTrue(all(value >= 0.0 for value in result.cost_history))

    def test_two_points_recover_a_linear_relationship(self):
        result = train_model(
            mileages=(100.0, 200.0),
            prices=(300.0, 100.0),
            iterations=1000,
        )

        self.assertAlmostEqual(result.theta0, 500.0, places=8)
        self.assertAlmostEqual(result.theta1, -2.0, places=8)

    def test_constant_prices_produce_zero_slope(self):
        result = train_model(
            mileages=(100.0, 200.0, 300.0),
            prices=(42.0, 42.0, 42.0),
            iterations=500,
        )

        self.assertAlmostEqual(result.theta0, 42.0, places=8)
        self.assertAlmostEqual(result.theta1, 0.0, places=8)

    def test_standardization_handles_large_values(self):
        result = train_model(
            mileages=(1_000_000_000.0, 1_200_000_000.0, 1_400_000_000.0),
            prices=(5_000_000.0, 4_600_000.0, 4_200_000.0),
            iterations=100,
        )

        values = (result.theta0, result.theta1, *result.cost_history)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertLess(result.cost_history[-1], result.cost_history[0])

    def test_converts_normalized_parameters_to_original_scale(self):
        result = train_model(self.MILEAGES, self.PRICES, iterations=500)

        for mileage in self.MILEAGES:
            normalized_km = (mileage - result.mean_km) / result.std_km
            normalized_prediction = estimate_price(
                result.normalized_theta0,
                result.normalized_theta1,
                normalized_km,
            )
            original_prediction = estimate_price(
                result.theta0,
                result.theta1,
                mileage,
            )
            self.assertAlmostEqual(original_prediction, normalized_prediction)

    def test_rejects_zero_mileage_standard_deviation(self):
        with self.assertRaisesRegex(ValueError, "standard deviation"):
            train_model((100.0, 100.0), (10.0, 20.0))

    def test_rejects_empty_training_data(self):
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            train_model((), ())

    def test_rejects_single_point_training_data(self):
        with self.assertRaisesRegex(ValueError, "standard deviation"):
            train_model((100.0,), (42.0,))

    def test_rejects_mismatched_training_columns(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            train_model((100.0,), (42.0, 43.0))

    def test_rejects_non_finite_training_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            train_model((100.0, float("nan")), (42.0, 43.0))

    def test_rejects_non_positive_learning_rate(self):
        with self.assertRaisesRegex(ValueError, "Learning rate"):
            train_model(self.MILEAGES, self.PRICES, learning_rate=0.0)

    def test_rejects_non_finite_learning_rate(self):
        with self.assertRaisesRegex(ValueError, "Learning rate"):
            train_model(self.MILEAGES, self.PRICES, learning_rate=float("inf"))

    def test_rejects_non_positive_iterations(self):
        with self.assertRaisesRegex(ValueError, "Iterations"):
            train_model(self.MILEAGES, self.PRICES, iterations=0)

    def test_rejects_unstable_learning_rate(self):
        with self.assertRaisesRegex(ValueError, "non-finite parameter"):
            train_model(self.MILEAGES, self.PRICES, learning_rate=1e308)

    def test_training_parameters_and_cost_remain_finite(self):
        result = train_model(self.MILEAGES, self.PRICES)

        values = (
            result.theta0,
            result.theta1,
            result.normalized_theta0,
            result.normalized_theta1,
            *result.cost_history,
        )
        self.assertTrue(all(math.isfinite(value) for value in values))


if __name__ == "__main__":
    unittest.main()
