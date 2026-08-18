import copy
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_macs_application import TRACE as MACS_TRACE
from scripts.run_macs_application import _tuning as macs_tuning
from scripts.run_strict_benchmark import TRACE, _default_tuning, registered_scenarios
from scripts.run_trace_tuning_calibration import (
    LAMBDA_RATIO_GRID,
    PILOT_DATA_SEEDS,
    PILOT_SCENARIOS,
    ROUGHNESS_GRID,
    _subject_balanced_huber_loss,
)
from src.trace_tuning_protocol import (
    DEFAULT_TRACE_TUNING_PATH,
    TUNING_MODE,
    load_trace_tuning_lock,
    validate_trace_tuning_payload,
)


class TraceTuningCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(DEFAULT_TRACE_TUNING_PATH)
        cls.payload = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.lock = load_trace_tuning_lock(cls.path)

    def test_locked_artifact_is_complete_and_hash_verified(self) -> None:
        self.assertEqual(len(self.payload["cohorts"]), 15)
        self.assertEqual(len(self.payload["evaluations"]), 270)
        self.assertEqual(len(self.payload["candidate_summaries"]), 18)
        self.assertEqual(self.payload["pilot_data_seeds"], list(PILOT_DATA_SEEDS))
        self.assertEqual(self.payload["pilot_scenarios"], list(PILOT_SCENARIOS))
        self.assertEqual(
            self.payload["candidate_grid"]["lambda_ratio"],
            list(LAMBDA_RATIO_GRID),
        )
        self.assertEqual(
            self.payload["candidate_grid"]["roughness"], list(ROUGHNESS_GRID)
        )
        self.assertTrue(
            all(item["status"] == "success" for item in self.payload["evaluations"])
        )
        self.assertEqual(len(str(self.lock["content_sha256"])), 64)
        self.assertEqual(len(str(self.lock["file_sha256"])), 64)

    def test_selected_pair_is_registered_global_minimizer(self) -> None:
        best = min(
            self.payload["candidate_summaries"],
            key=lambda item: (
                item["mean_validation_huber_loss"],
                item["lambda_ratio"],
                item["roughness"],
            ),
        )
        self.assertEqual(self.lock["lambda_ratio"], best["lambda_ratio"])
        self.assertEqual(self.lock["roughness"], best["roughness"])
        self.assertEqual(self.lock["tuning_mode"], TUNING_MODE)

    def test_pilot_seeds_are_audited_disjoint_from_formal_mc(self) -> None:
        audit = self.payload["seed_independence_audit"]
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["formal_data_seed_overlap"], [])
        self.assertEqual(audit["formal_split_seed_overlap"], [])
        self.assertEqual(len(set(PILOT_DATA_SEEDS)), 5)

    def test_validation_huber_loss_balances_subjects_not_rows(self) -> None:
        response = np.asarray([1.0, 1.0, 1.0, 3.0])
        prediction = np.zeros(4)
        subjects = np.asarray(["a", "a", "a", "b"])
        # With delta=10 this is squared loss/2. Subject a contributes 0.5,
        # subject b contributes 4.5, and the subject-balanced mean is 2.5.
        self.assertAlmostEqual(
            _subject_balanced_huber_loss(
                response, prediction, subjects, delta=10.0
            ),
            2.5,
        )

    def test_strict_and_macs_runners_use_the_same_locked_pair(self) -> None:
        scenario = next(
            item
            for item in registered_scenarios(quick=False)
            if item.scenario == "example2-gaussian-n50-sigma0.1"
        )
        strict_formal = _default_tuning(TRACE, scenario, quick=False)
        strict_quick = _default_tuning(TRACE, scenario, quick=True)
        macs_formal = macs_tuning(MACS_TRACE, 6, quick=False)
        macs_quick = macs_tuning(MACS_TRACE, 4, quick=True)
        for tuning in (strict_formal, strict_quick, macs_formal, macs_quick):
            self.assertEqual(tuning["lambda_ratio"], self.lock["lambda_ratio"])
            self.assertEqual(tuning["roughness"], self.lock["roughness"])
            self.assertEqual(tuning["tuning_mode"], TUNING_MODE)
            self.assertEqual(
                tuning["calibration_content_sha256"], self.lock["content_sha256"]
            )
            self.assertEqual(
                tuning["calibration_file_sha256"], self.lock["file_sha256"]
            )
        self.assertEqual(strict_formal["q_time"], 6)
        self.assertEqual(strict_formal["q_covariate"], 6)

    def test_tampering_is_rejected_by_content_hash(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["evaluations"][0]["validation_huber_loss"] += 1.0
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            validate_trace_tuning_payload(tampered)


if __name__ == "__main__":
    unittest.main()

