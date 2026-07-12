import unittest

import numpy as np
import scipy as sp

from util import reduce_by_svd


class ReduceBySvdTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260712)
        self.X = rng.normal(size=(10, 6))

    def test_helmert_shape_mean_covariance_and_input_unchanged(self):
        original = self.X.copy()
        m_reduced = 5

        reduced = reduce_by_svd(self.X, m_reduced)

        self.assertEqual(reduced.shape, (m_reduced, self.X.shape[1]))
        np.testing.assert_array_equal(self.X, original)
        np.testing.assert_allclose(reduced.mean(axis=0), self.X.mean(axis=0), atol=1e-14)

        anomalies = self.X - self.X.mean(axis=0)
        U, S, _ = sp.linalg.svd(anomalies.T)
        n_modes = m_reduced - 1
        expected_covariance = (
            (U[:, :n_modes] * (S[:n_modes] ** 2)) @ U[:, :n_modes].T
        ) / (self.X.shape[0] - 1)
        actual_covariance = np.cov(reduced, rowvar=False, ddof=1)
        np.testing.assert_allclose(actual_covariance, expected_covariance, atol=1e-14)

    def test_none_uses_legacy_principal_component_construction(self):
        m_reduced = 4
        reduced = reduce_by_svd(self.X, m_reduced, method=None)

        mean = self.X.mean(axis=0)
        U, S, _ = sp.linalg.svd((self.X - mean).T)
        expected = mean + (U[:, :m_reduced] * S[:m_reduced]).T
        np.testing.assert_allclose(reduced, expected, atol=1e-14)

    def test_minimum_and_unchanged_ensemble_sizes(self):
        for m_reduced in (2, self.X.shape[0]):
            with self.subTest(m_reduced=m_reduced):
                reduced = reduce_by_svd(self.X, m_reduced)
                self.assertEqual(reduced.shape, (m_reduced, self.X.shape[1]))
                np.testing.assert_allclose(reduced.mean(axis=0), self.X.mean(axis=0))

    def test_invalid_arguments(self):
        invalid_calls = (
            (self.X[0], 2, "helmert"),
            (self.X, 1, "helmert"),
            (self.X, self.X.shape[0] + 1, "helmert"),
            (self.X, 3.0, "helmert"),
            (self.X, True, "helmert"),
            (self.X, 3, "unknown"),
        )
        for X, m_reduced, method in invalid_calls:
            with self.subTest(m_reduced=m_reduced, method=method):
                with self.assertRaises(ValueError):
                    reduce_by_svd(X, m_reduced, method=method)


if __name__ == "__main__":
    unittest.main()
