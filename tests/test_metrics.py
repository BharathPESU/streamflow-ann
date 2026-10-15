import unittest
import numpy as np
from src.metrics import rmse, nse, kge, evaluate


class TestMetrics(unittest.TestCase):
    def test_rmse_perfect(self):
        a = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(rmse(a, a), 0.0)

    def test_rmse_known_value(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        self.assertAlmostEqual(rmse(a, b), 1.0)

    def test_nse_perfect_score(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(nse(y_true, y_true), 1.0)

    def test_nse_mean_prediction(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.full_like(y_true, np.mean(y_true))
        self.assertAlmostEqual(nse(y_true, y_pred), 0.0)

    def test_kge_perfect_score(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(kge(y_true, y_true), 1.0)

    def test_evaluate_keys(self):
        y_true = np.array([0.1, 0.2, 0.6])
        y_pred = np.array([0.12, 0.19, 0.58])
        res = evaluate(y_true, y_pred, q_max=3000.0)
        self.assertIn("nse_overall", res)
        self.assertIn("kge_overall", res)
        self.assertIn("peak", res)
        self.assertIn("nonpeak", res)


if __name__ == "__main__":
    unittest.main()
