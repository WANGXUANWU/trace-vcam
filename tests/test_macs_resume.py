import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.adapters.base import PreflightReport
from benchmarks.methods import FIXED_METHOD_LABELS, METHOD_SPECS
from scripts import run_macs_application as macs


class _PreflightOnlyAdapter:
    def __init__(self, label: str) -> None:
        self.label = label

    def preflight(self) -> PreflightReport:
        return PreflightReport(
            True,
            "macs-resume-fixture/1",
            environment={"fixture": "stable"},
        )


def _fixture_registry() -> dict[str, _PreflightOnlyAdapter]:
    return {method: _PreflightOnlyAdapter(method) for method in FIXED_METHOD_LABELS}


def _fixture_raw() -> macs.RawMACS:
    person = np.asarray(["1", "1", "2", "2", "3", "3", "4", "4"], dtype=str)
    return macs.RawMACS(
        cd4=np.asarray([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]),
        time=np.asarray([-1.0, -0.5, -1.0, -0.5, -1.0, -0.5, -1.0, -0.5]),
        age=np.asarray([-2.0, -2.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0]),
        cesd=np.asarray([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]),
        person=person,
        row_id=np.asarray([f"macs-row-{index}" for index in range(len(person))], dtype=str),
    )


def _fixture_cohort_payload(task):
    cohort, mode, _quick, admissions, _preflight = task
    train = cohort.dataset.subset_subjects(cohort.split.train_subjects)
    test = cohort.dataset.subset_subjects(cohort.split.test_subjects)
    rows = []
    predictions = []
    for method in FIXED_METHOD_LABELS:
        applicability, reason = macs._macs_applicability(method)
        admission = "not_required" if method == macs.TRACE else admissions.get(
            method, "pending_no_reproduction"
        )
        if applicability == "N/A by design":
            status = "N/A by design"
            failure_code = "not_applicable"
        elif method != macs.TRACE and admission != "admitted":
            status = "blocked_reproduction_gate"
            failure_code = "reproduction_not_verified"
        else:
            status = "success"
            failure_code = ""
        row = {field: "" for field in macs.RESULT_FIELDS}
        row.update(
            schema_version=macs.SCHEMA_VERSION,
            mode=mode,
            variant=cohort.variant,
            basis_dimension=cohort.basis_dimension,
            repeat=cohort.split.repeat,
            fold=cohort.split.fold,
            fold_seed=cohort.split.seed,
            method=method,
            method_display_name=METHOD_SPECS[method].display_name,
            method_version="macs-resume-fixture/1",
            applicability=applicability,
            applicability_reason=reason,
            admission_status=admission,
            attempt_status=status,
            converged=status == "success",
            failure_code=failure_code,
            failure_message="",
            n_subjects=cohort.dataset.n_subjects,
            n_rows=cohort.dataset.n_rows,
            n_train_subjects=train.n_subjects,
            n_test_subjects=test.n_subjects,
            data_hash=cohort.dataset.data_hash,
            train_subject_hash=cohort.split.train_hash,
            test_subject_hash=cohort.split.test_hash,
            tuning_json="{}",
            realized_tuning_json="{}",
            runtime_seconds=0.01 if status == "success" else float("nan"),
            peak_python_memory_mb=0.1 if status == "success" else float("nan"),
            test_mse=0.0 if status == "success" else float("nan"),
            subject_balanced_test_mse=0.0 if status == "success" else float("nan"),
            test_mae=0.0 if status == "success" else float("nan"),
            fit_metadata_json="{}",
        )
        rows.append(row)
        if status == "success":
            for index in np.argsort(test.row_id.astype(str)):
                predictions.append(
                    {
                        "schema_version": macs.SCHEMA_VERSION,
                        "variant": cohort.variant,
                        "basis_dimension": cohort.basis_dimension,
                        "repeat": cohort.split.repeat,
                        "fold": cohort.split.fold,
                        "method": method,
                        "row_id": str(test.row_id[index]),
                        "subject_id": str(test.subject_id[index]),
                        "observed_cd4": float(test.response[index]),
                        "prediction": float(test.response[index]),
                    }
                )
    return cohort.index, rows, predictions


def _fixture_curve(_adapter, *, dataset, seed, **_kwargs):
    curves = [
        {
            "component": "baseline",
            "domain": "time",
            "grid": [0.0, 1.0],
            "values": [0.0, 0.0],
        }
    ]
    audit = {
        "fit_id": "macs-primary-full-data-trace-v1",
        "attempt_status": "success",
        "converged": True,
        "data_hash": dataset.data_hash,
        "tuning_sha256": "a" * 64,
        "raw_curves_sha256": macs._object_sha256(curves),
        "identified_curves_sha256": macs._object_sha256(curves),
    }
    row = {
        "schema_version": macs.SCHEMA_VERSION,
        "variant": "primary",
        "basis_dimension": 4,
        "repeat": 0,
        "fold": 0,
        "repeat_fold_semantics": "compatibility selector; not a CV fit",
        "method": macs.TRACE,
        "fit_id": audit["fit_id"],
        "fit_scope": "registered_full_data",
        "seed": int(seed),
        "n_subjects": dataset.n_subjects,
        "n_rows": dataset.n_rows,
        "data_hash": dataset.data_hash,
        "method_version": "macs-resume-fixture/1",
        "tuning_sha256": audit["tuning_sha256"],
        "identified_curves_sha256": audit["identified_curves_sha256"],
        "curves": curves,
    }
    audit["curve_row_sha256"] = macs._object_sha256(row)
    return row, audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MACSResumeTests(unittest.TestCase):
    def _args(self, output: Path, *, jobs: int = 1):
        return macs.parse_args(["--quick", "--output", str(output), "--jobs", str(jobs)])

    def _patches(self):
        return (
            patch.object(macs, "read_macs_csv", return_value=_fixture_raw()),
            patch.object(macs, "adapter_registry", side_effect=_fixture_registry),
            patch.object(
                macs,
                "_fit_registered_full_data_trace_curves",
                side_effect=_fixture_curve,
            ),
        )

    def test_parallel_prefetch_is_bounded_and_ordered(self):
        class FakeFuture:
            def __init__(self, owner, function, args):
                self.owner = owner
                self.function = function
                self.args = args
                self.consumed = False

            def result(self):
                if not self.consumed:
                    self.consumed = True
                    self.owner.outstanding -= 1
                return self.function(*self.args)

        class FakeExecutor:
            instance = None

            def __init__(self, max_workers):
                self.max_workers = max_workers
                self.outstanding = 0
                self.maximum_outstanding = 0
                FakeExecutor.instance = self

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, function, *args):
                self.outstanding += 1
                self.maximum_outstanding = max(
                    self.maximum_outstanding, self.outstanding
                )
                return FakeFuture(self, function, args)

        jobs = 3
        tasks = [(index,) for index in range(14)]
        with patch.object(
            macs.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
        ):
            observed = list(
                macs._ordered_macs_cohort_results(
                    tasks,
                    jobs=jobs,
                    worker=lambda task: (task[0], [], []),
                )
            )
        executor = FakeExecutor.instance
        self.assertEqual([item[0] for item in observed], list(range(14)))
        self.assertEqual(executor.maximum_outstanding, macs._macs_prefetch_limit(jobs))
        self.assertLessEqual(
            executor.maximum_outstanding, macs.ORDERED_TASK_PREFETCH_FACTOR * jobs
        )
        self.assertEqual(executor.outstanding, 0)

    def test_interrupted_tail_is_repaired_resumed_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "macs"
            args = self._args(output)
            first_calls = []

            def interrupted(task):
                index = task[0].index
                first_calls.append(index)
                if index == 1:
                    raise RuntimeError("synthetic MACS interruption")
                return _fixture_cohort_payload(task)

            patches = self._patches()
            with patches[0], patches[1], patches[2], patch.object(
                macs, "_macs_cohort_task", side_effect=interrupted
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic MACS interruption"):
                    macs.execute(args)
            self.assertEqual(first_calls, [0, 1])
            progress_path = output / "macs_progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["committed_cohorts"], 1)
            self.assertEqual(progress["schema_version"], macs.PROGRESS_SCHEMA_VERSION)
            self.assertEqual(progress["curve_completion"]["state"], "pending")
            for key, filename in {
                "results": "macs_results.csv",
                "predictions": "macs_predictions.csv",
                "curves": "macs_factor_curves.jsonl",
            }.items():
                path = output / filename
                with path.open("ab") as handle:
                    handle.write(b"UNCOMMITTED-TAIL")
                self.assertGreater(
                    path.stat().st_size, int(progress["committed_offsets"][key])
                )

            resumed_calls = []

            def resumed(task):
                resumed_calls.append(task[0].index)
                return _fixture_cohort_payload(task)

            patches = self._patches()
            with patches[0], patches[1], patches[2], patch.object(
                macs, "_macs_cohort_task", side_effect=resumed
            ):
                paths = macs.execute(args)
            self.assertEqual(resumed_calls, [1])
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(final_progress["committed_cohorts"], 2)
            self.assertEqual(final_progress["curve_completion"]["state"], "completed")
            self.assertEqual(final_progress["status"], "finalized")
            for key in ("results", "predictions", "curves"):
                self.assertNotIn(b"UNCOMMITTED-TAIL", paths[key].read_bytes())
                self.assertEqual(
                    paths[key].stat().st_size,
                    int(final_progress["committed_offsets"][key]),
                )
                self.assertEqual(_sha256(paths[key]), final_progress["committed_sha256"][key])
            with paths["results"].open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2 * len(FIXED_METHOD_LABELS))

            hashes_before = {key: _sha256(path) for key, path in paths.items()}

            def forbidden(*_args, **_kwargs):
                raise AssertionError("an idempotent MACS resume must not rerun work")

            patches = self._patches()
            with patches[0], patches[1], patches[2], patch.object(
                macs, "_macs_cohort_task", side_effect=forbidden
            ), patch.object(
                macs, "_fit_registered_full_data_trace_curves", side_effect=forbidden
            ):
                paths_again = macs.execute(args)
            hashes_after = {key: _sha256(path) for key, path in paths_again.items()}
            self.assertEqual(hashes_after, hashes_before)

    def test_resume_refuses_tampered_committed_prefix_and_changed_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "macs"
            args = self._args(output)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patch.object(
                macs, "_macs_cohort_task", side_effect=_fixture_cohort_payload
            ):
                paths = macs.execute(args)

            changed_jobs = self._args(output, jobs=2)
            patches = self._patches()
            with patches[0], patches[1], patches[2]:
                with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                    macs.execute(changed_jobs)

            result_bytes = bytearray(paths["results"].read_bytes())
            result_bytes[-2] = ord("X") if result_bytes[-2] != ord("X") else ord("Y")
            paths["results"].write_bytes(result_bytes)
            patches = self._patches()
            with patches[0], patches[1], patches[2]:
                with self.assertRaisesRegex(RuntimeError, "prefix hash mismatch"):
                    macs.execute(args)

    def test_curve_fit_is_resumed_after_all_folds_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "macs"
            args = self._args(output)
            worker_calls = []

            def worker(task):
                worker_calls.append(task[0].index)
                return _fixture_cohort_payload(task)

            patches = self._patches()
            with patches[0], patches[1], patch.object(
                macs, "_macs_cohort_task", side_effect=worker
            ), patch.object(
                macs,
                "_fit_registered_full_data_trace_curves",
                side_effect=RuntimeError("synthetic curve interruption"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic curve interruption"):
                    macs.execute(args)
            self.assertEqual(worker_calls, [0, 1])
            progress_path = output / "macs_progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["committed_cohorts"], 2)
            self.assertEqual(progress["curve_completion"]["state"], "pending")

            def forbidden_worker(*_args, **_kwargs):
                raise AssertionError("committed folds must not rerun for a curve resume")

            patches = self._patches()
            with patches[0], patches[1], patches[2], patch.object(
                macs, "_macs_cohort_task", side_effect=forbidden_worker
            ):
                macs.execute(args)
            final = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "finalized")
            self.assertEqual(final["curve_completion"]["state"], "completed")

    def test_final_cohort_audit_normalizes_csv_keys_and_refinalizes_metadata_only(self):
        """A completed CSV uses string repeat/fold fields but still has shared splits."""

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "macs"
            args = self._args(output)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patch.object(
                macs, "_macs_cohort_task", side_effect=_fixture_cohort_payload
            ):
                paths = macs.execute(args)

            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertTrue(metadata["cohort_audit"]["passed"])
            self.assertEqual(metadata["cohort_audit"]["expected_cohorts"], 2)
            self.assertEqual(metadata["cohort_audit"]["observed_cohorts"], 2)
            self.assertEqual(metadata["cohort_audit"]["observed_result_rows"], 12)

            # Model a finalized historical metadata artifact made by the old
            # string-versus-int final audit.  The raw files themselves remain
            # byte-identical and are the evidence the migration must preserve.
            legacy = dict(metadata)
            legacy["cohort_audit"] = {
                "passed": False,
                "issues": ["primary/0/0: method cohort mismatch"],
            }
            legacy["formal_protocol_complete"] = False
            legacy["descriptive_results_eligible"] = False
            legacy["formal_claims_eligible"] = False
            legacy["formal_claims_reason"] = "legacy final audit false negative"
            legacy_bytes = (
                json.dumps(legacy, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            paths["metadata"].write_bytes(legacy_bytes)
            legacy_hash = hashlib.sha256(legacy_bytes).hexdigest()
            paths["metadata_sha256"].write_text(
                f"{legacy_hash}  {paths['metadata'].name}\n", encoding="ascii"
            )
            raw_before = {
                key: _sha256(paths[key]) for key in ("results", "predictions", "curves")
            }

            repaired = macs.refinalize_existing_macs_metadata(output)
            raw_after = {
                key: _sha256(paths[key]) for key in ("results", "predictions", "curves")
            }
            self.assertEqual(raw_after, raw_before)
            self.assertEqual(repaired["snapshot"].read_bytes(), legacy_bytes)
            repaired_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertTrue(repaired_metadata["cohort_audit"]["passed"])
            self.assertEqual(repaired_metadata["cohort_audit"]["issues"], [])
            self.assertEqual(
                repaired_metadata["metadata_refinalizations"][-1]["previous_metadata_sha256"],
                legacy_hash,
            )
            sidecar = json.loads(repaired["audit"].read_text(encoding="utf-8"))
            self.assertEqual(sidecar["state"], "completed")
            self.assertEqual(sidecar["raw_streams_before"], sidecar["raw_streams_after"])
            self.assertEqual(
                paths["metadata_sha256"].read_text(encoding="ascii").split()[0],
                _sha256(paths["metadata"]),
            )

            # The repair command is idempotent and does not rewrite the final
            # metadata after its audited sidecar has been published.
            repaired_hash = _sha256(paths["metadata"])
            macs.refinalize_existing_macs_metadata(output)
            self.assertEqual(_sha256(paths["metadata"]), repaired_hash)


if __name__ == "__main__":
    unittest.main()
