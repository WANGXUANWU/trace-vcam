import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.adapters.base import PreflightReport
from benchmarks.methods import FIXED_METHOD_LABELS, Protocol
from scripts import run_strict_benchmark as strict


class _PreflightOnlyAdapter:
    def __init__(self, label: str) -> None:
        self.label = label

    def preflight(self) -> PreflightReport:
        return PreflightReport(
            True,
            "resume-fixture/1",
            environment={"fixture": "stable"},
        )


def _fixture_registry():
    return (
        strict.Scenario(
            scenario="resume-fixture",
            phase="common",
            example="Fixture",
            protocol=Protocol.EXAMPLE2_GAUSSIAN.value,
            generator="zzw2020",
            parameters={
                "n_subjects": 4,
                "sigma": 0.1,
                "error_distribution": "gaussian",
            },
            formal_replications=2,
        ),
    )


def _fixture_adapters():
    return {method: _PreflightOnlyAdapter(method) for method in FIXED_METHOD_LABELS}


def _cohort_payload(scenario, replicate, root_seed, mode, quick, gates):
    del quick, gates
    seed = strict._stable_seed(root_seed, scenario.scenario, replicate, "data")
    split_seed = strict._stable_seed(
        root_seed, scenario.scenario, replicate, "subject-split"
    )
    triples = []
    for method in FIXED_METHOD_LABELS:
        success = method == strict.TRACE
        row = {field: "" for field in strict.RESULT_FIELDS}
        row.update(
            schema_version=strict.SCHEMA_VERSION,
            mode=mode,
            phase=scenario.phase,
            example=scenario.example,
            protocol=scenario.protocol,
            scenario=scenario.scenario,
            replicate=replicate,
            seed=seed,
            split_seed=split_seed,
            split_unit="subject",
            method=method,
            method_version="resume-fixture/1",
            applicability="applicable" if success else "N/A by design",
            applicability_reason="fixture",
            admission_status="not_required" if success else "admitted",
            attempt_status="success" if success else "N/A by design",
            converged=success,
            failure_code="" if success else "not_applicable",
            failure_message="",
            design_id="fixture-design",
            provenance="fixture",
            n_subjects=4,
            n_train_subjects=3,
            n_test_subjects=1,
            n_rows=8,
            n_covariates=2,
            n_active=2,
            has_null_blocks=False,
            data_hash="d" * 64,
            train_subject_hash="a" * 64,
            test_subject_hash="b" * 64,
            tuning_json="{}" if success else "",
            tuning_sha256="c" * 64 if success else "",
            runtime_seconds=0.01 if success else 0.0,
            peak_python_memory_mb=0.1 if success else 0.0,
            observed_test_mspe=0.0 if success else float("nan"),
            test_mse=0.0 if success else float("nan"),
            noise_free_test_mspe=0.0 if success else float("nan"),
            baseline_ise=0.0 if success else float("nan"),
            component_ise=0.0 if success else float("nan"),
            factor_ise=0.0 if success else float("nan"),
            paper_observed_factor_mse_total=float("nan"),
            paper_training_function_mse_total=float("nan"),
            tpr=float("nan"),
            fdr=float("nan"),
            model_size=float("nan"),
            selected_blocks_json="[0,1]" if success else "[]",
            fit_metadata_json="{}",
        )
        predictions = []
        curve = None
        if success:
            predictions = [
                {
                    "schema_version": strict.SCHEMA_VERSION,
                    "scenario": scenario.scenario,
                    "replicate": replicate,
                    "seed": seed,
                    "method": method,
                    "row_id": f"row-{replicate}-{index}",
                    "subject_id": f"subject-{replicate}",
                    "observed_response": float(index),
                    "noise_free_target": float(index),
                    "prediction": float(index),
                }
                for index in range(2)
            ]
            curve = {
                "schema_version": strict.SCHEMA_VERSION,
                "scenario": scenario.scenario,
                "replicate": replicate,
                "seed": seed,
                "method": method,
                "curves": [
                    {
                        "component": "baseline",
                        "domain": "time",
                        "grid": [0.0, 1.0],
                        "values": [0.0, 0.0],
                    }
                ],
            }
        triples.append((row, predictions, curve))
    return triples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrictResumeTests(unittest.TestCase):
    def test_parallel_window_is_bounded_and_yields_registered_order(self):
        scenario = _fixture_registry()[0]

        class FakeFuture:
            def __init__(self, owner, function, args):
                self.owner = owner
                self.function = function
                self.args = args
                self.replicate = int(args[0])
                self.remaining = 8 if self.replicate == 0 else 1
                self.started = False
                self.done = False
                self.consumed = False
                self.value = None

            def result(self):
                self.owner.advance_until(self)
                if not self.consumed:
                    self.consumed = True
                    self.owner.outstanding -= 1
                return self.value

        class FakeExecutor:
            instance = None

            def __init__(self, max_workers):
                self.max_workers = max_workers
                self.outstanding = 0
                self.maximum_outstanding = 0
                self.maximum_active = 0
                self.maximum_buffered = 0
                self.waiting = []
                self.active = []
                self.submissions = []
                self.completion_order = []
                FakeExecutor.instance = self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, function, *args):
                future = FakeFuture(self, function, args)
                self.outstanding += 1
                self.maximum_outstanding = max(
                    self.maximum_outstanding, self.outstanding
                )
                self.waiting.append(future)
                self.submissions.append(future)
                return future

            def _fill_workers(self):
                while self.waiting and len(self.active) < self.max_workers:
                    future = self.waiting.pop(0)
                    future.started = True
                    self.active.append(future)
                self.maximum_active = max(self.maximum_active, len(self.active))

            def _record_buffered(self):
                buffered = sum(
                    future.done and not future.consumed
                    for future in self.submissions
                )
                self.maximum_buffered = max(self.maximum_buffered, buffered)

            def advance_until(self, target):
                while not target.done:
                    self._fill_workers()
                    if not self.active:
                        raise AssertionError("target was never submitted")
                    completed = []
                    for future in self.active:
                        future.remaining -= 1
                        if future.remaining == 0:
                            future.value = future.function(*future.args)
                            future.done = True
                            completed.append(future)
                            self.completion_order.append(future.replicate)
                    self.active = [
                        future for future in self.active if future not in completed
                    ]
                    self._record_buffered()
                    self._fill_workers()

        jobs = 3
        limit = strict._ordered_task_prefetch_limit(jobs)
        self.assertEqual(strict._ordered_task_prefetch_limit(6), 18)
        tasks = [(None, scenario, replicate) for replicate in range(14)]
        with patch.object(
            strict.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
        ):
            observed = list(
                strict._ordered_task_results(
                    tasks,
                    jobs=jobs,
                    worker=lambda replicate: replicate,
                    worker_arguments=lambda item, index: (index,),
                )
            )
        executor = FakeExecutor.instance
        self.assertEqual([item[1] for item in observed], list(range(14)))
        self.assertEqual([item[2] for item in observed], list(range(14)))
        self.assertEqual(executor.maximum_active, jobs)
        self.assertLessEqual(executor.maximum_active, jobs)
        self.assertEqual(executor.maximum_outstanding, limit)
        self.assertLessEqual(executor.maximum_buffered, limit)
        self.assertEqual(executor.outstanding, 0)
        completed_before_slow_head = executor.completion_order[
            : executor.completion_order.index(0)
        ]
        self.assertGreater(len(completed_before_slow_head), jobs - 1)
        self.assertIn(limit - 1, completed_before_slow_head)

    def test_resume_contract_locks_jobs_and_prefetch_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "strict"
            original_args = strict.parse_args(
                [
                    "--quick",
                    "--output",
                    str(output),
                    "--jobs",
                    "1",
                    "--quick-replications",
                    "2",
                ]
            )
            with patch.object(
                strict, "registered_scenarios", return_value=_fixture_registry()
            ), patch.object(
                strict, "adapter_registry", side_effect=_fixture_adapters
            ), patch.object(strict, "_common_task", side_effect=_cohort_payload):
                paths = strict.execute(original_args)

            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
            expected_execution = {
                "jobs": 1,
                "ordered_task_scheduler_version": (
                    strict.ORDERED_TASK_SCHEDULER_VERSION
                ),
                "ordered_task_prefetch_factor": (
                    strict.ORDERED_TASK_PREFETCH_FACTOR
                ),
                "max_outstanding_futures": 1,
            }
            self.assertEqual(progress["run_contract"]["execution"], expected_execution)
            self.assertEqual(metadata["execution"]["jobs"], 1)
            self.assertEqual(
                metadata["execution"]["ordered_task_prefetch_factor"],
                strict.ORDERED_TASK_PREFETCH_FACTOR,
            )
            self.assertEqual(metadata["execution"]["max_outstanding_futures"], 1)
            self.assertEqual(
                metadata["execution"]["max_parent_buffered_cohort_payloads"], 1
            )

            changed_jobs_args = strict.parse_args(
                [
                    "--quick",
                    "--output",
                    str(output),
                    "--jobs",
                    "2",
                    "--quick-replications",
                    "2",
                ]
            )
            with patch.object(
                strict, "registered_scenarios", return_value=_fixture_registry()
            ), patch.object(
                strict, "adapter_registry", side_effect=_fixture_adapters
            ), patch.object(strict, "_common_task", side_effect=_cohort_payload):
                with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                    strict.execute(changed_jobs_args)

            with patch.object(
                strict,
                "ORDERED_TASK_PREFETCH_FACTOR",
                strict.ORDERED_TASK_PREFETCH_FACTOR + 1,
            ), patch.object(
                strict, "registered_scenarios", return_value=_fixture_registry()
            ), patch.object(
                strict, "adapter_registry", side_effect=_fixture_adapters
            ), patch.object(strict, "_common_task", side_effect=_cohort_payload):
                with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                    strict.execute(original_args)

    def test_interrupted_tail_is_truncated_resumed_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "strict"
            args = strict.parse_args(
                [
                    "--quick",
                    "--output",
                    str(output),
                    "--jobs",
                    "1",
                    "--quick-replications",
                    "2",
                ]
            )
            first_calls = []

            def interrupted(*worker_args):
                replicate = int(worker_args[1])
                first_calls.append(replicate)
                if replicate == 1:
                    raise RuntimeError("synthetic interruption")
                return _cohort_payload(*worker_args)

            common_patches = (
                patch.object(strict, "registered_scenarios", return_value=_fixture_registry()),
                patch.object(strict, "adapter_registry", side_effect=_fixture_adapters),
            )
            with common_patches[0], common_patches[1], patch.object(
                strict, "_common_task", side_effect=interrupted
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    strict.execute(args)
            self.assertEqual(first_calls, [0, 1])
            progress_path = output / "strict_progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["committed_cohorts"], 1)
            self.assertEqual(progress["schema_version"], strict.PROGRESS_SCHEMA_VERSION)
            for key, filename in {
                "results": "strict_results.csv",
                "predictions": "strict_predictions.csv",
                "curves": "strict_factor_curves.jsonl",
            }.items():
                path = output / filename
                with path.open("ab") as handle:
                    handle.write(b"UNCOMMITTED-TAIL")
                self.assertGreater(
                    path.stat().st_size, int(progress["committed_offsets"][key])
                )

            resumed_calls = []

            def resumed(*worker_args):
                resumed_calls.append(int(worker_args[1]))
                return _cohort_payload(*worker_args)

            with patch.object(
                strict, "registered_scenarios", return_value=_fixture_registry()
            ), patch.object(
                strict, "adapter_registry", side_effect=_fixture_adapters
            ), patch.object(strict, "_common_task", side_effect=resumed):
                paths = strict.execute(args)
            self.assertEqual(resumed_calls, [1])
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(final_progress["committed_cohorts"], 2)
            self.assertEqual(final_progress["status"], "finalized")
            for key in ("results", "predictions", "curves"):
                self.assertNotIn(b"UNCOMMITTED-TAIL", paths[key].read_bytes())
                self.assertEqual(
                    paths[key].stat().st_size,
                    int(final_progress["committed_offsets"][key]),
                )
                self.assertEqual(
                    _sha256(paths[key]), final_progress["committed_sha256"][key]
                )
            with paths["results"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2 * len(FIXED_METHOD_LABELS))
            hashes_before = {
                key: _sha256(path)
                for key, path in paths.items()
                if key != "progress"
            }

            def forbidden_worker(*args, **kwargs):
                raise AssertionError("an idempotent resume must not rerun a cohort")

            with patch.object(
                strict, "registered_scenarios", return_value=_fixture_registry()
            ), patch.object(
                strict, "adapter_registry", side_effect=_fixture_adapters
            ), patch.object(strict, "_common_task", side_effect=forbidden_worker):
                paths_again = strict.execute(args)
            hashes_after = {
                key: _sha256(path)
                for key, path in paths_again.items()
                if key != "progress"
            }
            self.assertEqual(hashes_after, hashes_before)


if __name__ == "__main__":
    unittest.main()
