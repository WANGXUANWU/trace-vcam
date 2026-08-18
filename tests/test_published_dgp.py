import unittest

import numpy as np

from experiments.dgp import (
    generate_hhy2021,
    generate_zsy2026,
    generate_zw2015,
    generate_zzw2020,
    subject_split,
    zzw2020_truth,
)


class PublishedDGPTests(unittest.TestCase):
    def test_zw2015_dimensions_and_time_invariance(self):
        data = generate_zw2015(17)
        self.assertEqual(data.n_subjects, 100)
        self.assertEqual(data.time.size, 4000)
        for subject in range(100):
            rows = data.subject == subject
            self.assertTrue(np.all(data.covariates[rows] == data.covariates[rows][0]))

    def test_zzw_truth_obeys_identification_constraints(self):
        truth = zzw2020_truth()
        time = np.linspace(0.0, 2.0, 20001)
        z = np.linspace(0.0, 1.0, 20001)
        for beta in truth.beta:
            self.assertAlmostEqual(float(np.trapezoid(beta(time), time) / 2.0), 1.0, places=6)
        for phi in truth.phi:
            self.assertAlmostEqual(float(np.trapezoid(phi(z), z)), 0.0, places=6)

    def test_zzw_longitudinal_support_and_provenance(self):
        original = generate_zzw2020(19, n_subjects=50, sigma=0.4)
        extension = generate_zzw2020(
            19, n_subjects=50, sigma=0.4, error_distribution="hhy-mixed-normal"
        )
        self.assertEqual(original.provenance, "original")
        self.assertEqual(extension.provenance, "robustness-extension")
        self.assertTrue(np.all((original.covariates >= 0.0) & (original.covariates <= 1.0)))

    def test_hhy_original_size(self):
        data = generate_hhy2021(23, error_distribution="hhy-t2")
        self.assertEqual(data.n_subjects, 30)
        self.assertEqual(data.time.size, 600)
        self.assertEqual(data.provenance, "original")
        # The paper centers by the population E[phi_raw(X)], not the realized
        # sample average.  Since phi_raw(0)=0, this reads the deterministic
        # registered centering constant directly from the truth function.
        center = -float(data.truth.phi[0](np.asarray([0.0]))[0])
        self.assertAlmostEqual(center, 0.7131078461795729, places=12)

    def test_zsy_original_has_ten_active_blocks(self):
        data = generate_zsy2026(29, n_subjects=50, sigma=0.1)
        self.assertEqual(data.covariates.shape[1], 10)
        self.assertEqual(sum(data.truth.active), 10)
        self.assertEqual(data.provenance, "original")
        scaled = generate_zsy2026(29, n_subjects=50, sigma=0.1, n_covariates=25)
        self.assertEqual(sum(scaled.truth.active), 10)
        self.assertNotEqual(scaled.provenance, "original")

    def test_subject_split_has_no_leakage(self):
        data = generate_zzw2020(31, n_subjects=50, sigma=0.1)
        train, test = subject_split(data.subject, seed=37)
        self.assertEqual(np.intersect1d(data.subject[train], data.subject[test]).size, 0)
        self.assertEqual(np.union1d(train, test).size, data.time.size)


if __name__ == "__main__":
    unittest.main()
