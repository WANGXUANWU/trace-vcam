from __future__ import annotations

import json
import unittest

from scripts.audit_extreme_finite_success import flag_extreme_finite_successes


class ExtremeFiniteSuccessAuditTests(unittest.TestCase):
    def test_flags_extreme_finite_success_without_reclassifying_it(self) -> None:
        row = {
            "method": "ZZW2020",
            "attempt_status": "success",
            "converged": "True",
            "scenario": "example3-gaussian-n50-p10-sigma0.4",
            "replicate": "36",
            "component_ise": "7067.912656300403",
            "factor_ise": "3262.0442712279855",
            "fit_metadata_json": json.dumps({"numerical_mrs_increase": True}),
        }
        flags, summary = flag_extreme_finite_successes([row])

        self.assertEqual(row["attempt_status"], "success")
        self.assertEqual(row["converged"], "True")
        self.assertEqual(summary["candidate_successful_converged_rows"], 1)
        self.assertEqual(len(flags), 1)
        self.assertTrue(flags[0]["extreme_finite_success_audit_flag"])
        self.assertTrue(flags[0]["metadata_numerical_mrs_increase"])
        self.assertEqual(
            json.loads(str(flags[0]["flagged_metrics_json"])),
            {"component_ise": 7067.912656300403, "factor_ise": 3262.0442712279855},
        )

    def test_does_not_flag_nonfinite_or_non_success_rows(self) -> None:
        rows = [
            {
                "method": "ZZW2020",
                "attempt_status": "failed",
                "converged": "False",
                "scenario": "example3-gaussian-n50-p10-sigma0.4",
                "component_ise": "999999",
            },
            {
                "method": "ZZW2020",
                "attempt_status": "success",
                "converged": "True",
                "scenario": "example3-gaussian-n50-p10-sigma0.4",
                "component_ise": "nan",
            },
        ]
        flags, summary = flag_extreme_finite_successes(rows)

        self.assertEqual(flags, [])
        self.assertEqual(summary["candidate_successful_converged_rows"], 1)

    def test_default_scope_covers_every_observed_method_without_cross_method_pooling(self) -> None:
        rows = [
            {
                "method": "TRACE-VCAM",
                "attempt_status": "success",
                "converged": "True",
                "scenario": "example3-gaussian-n50-p10-sigma0.4",
                "replicate": "0",
                "component_ise": "0.2",
            },
            {
                "method": "ZZW2020",
                "attempt_status": "success",
                "converged": "True",
                "scenario": "example3-gaussian-n50-p10-sigma0.4",
                "replicate": "0",
                "component_ise": "1500",
            },
        ]
        flags, summary = flag_extreme_finite_successes(rows)

        self.assertEqual(set(summary["scope"]["methods"]), {"TRACE-VCAM", "ZZW2020"})
        self.assertEqual(summary["candidate_successful_converged_rows_by_method"], {
            "TRACE-VCAM": 1,
            "ZZW2020": 1,
        })
        self.assertEqual(summary["flagged_rows_by_method"], {"TRACE-VCAM": 0, "ZZW2020": 1})
        self.assertEqual([row["method"] for row in flags], ["ZZW2020"])


if __name__ == "__main__":
    unittest.main()
