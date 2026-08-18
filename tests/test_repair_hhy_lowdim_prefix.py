import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.adapters.base import PreflightReport
from benchmarks.methods import FIXED_METHOD_LABELS, METHOD_SPECS, Protocol
from scripts import repair_hhy_lowdim_prefix as repair
from scripts import run_strict_benchmark as strict


def _registry(*, example2_replications=2):
    return (
        strict.Scenario(
            scenario="repair-example1",
            phase="common",
            example="Example 1",
            protocol=Protocol.EXAMPLE1_DENSE.value,
            generator="zzw2020",
            parameters={"n_subjects": 4, "sigma": 0.1, "error_distribution": "gaussian"},
            formal_replications=1,
        ),
        strict.Scenario(
            scenario="repair-example2",
            phase="common",
            example="Example 2",
            protocol=Protocol.EXAMPLE2_GAUSSIAN.value,
            generator="zzw2020",
            parameters={"n_subjects": 4, "sigma": 0.1, "error_distribution": "gaussian"},
            formal_replications=example2_replications,
        ),
        strict.Scenario(
            scenario="repair-example3",
            phase="common",
            example="Example 3",
            protocol=Protocol.EXAMPLE3_HIGH_DIMENSIONAL.value,
            generator="zsy2026",
            parameters={"n_subjects": 4, "sigma": 0.1, "n_covariates": 10},
            formal_replications=1,
        ),
    )


def _tasks(*, example2_replications=2):
    scenarios = _registry(example2_replications=example2_replications)
    return [
        (scenario, replicate)
        for scenario in scenarios
        for replicate in range(scenario.formal_replications)
    ]


def _contract(*, jobs=1):
    return {
        "schema_version": strict.RUN_FINGERPRINT_SCHEMA_VERSION,
        "mode": "formal",
        "root_seed": strict.DEFAULT_ROOT_SEED,
        "execution": {"jobs": jobs, "fixture": "repair"},
        "method_order": list(FIXED_METHOD_LABELS),
        "source_sha256": {"benchmarks/adapters/hhy2021.py": "current-hhy"},
        "preflight": {strict.HHY: {"ready": True, "version": "fixture-hhy/2"}},
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cohort_identity(scenario, replicate):
    seed = strict._stable_seed(
        strict.DEFAULT_ROOT_SEED, scenario.scenario, replicate, "data"
    )
    split_seed = strict._stable_seed(
        strict.DEFAULT_ROOT_SEED, scenario.scenario, replicate, "subject-split"
    )
    return {
        "seed": seed,
        "split_seed": split_seed,
        "data_hash": f"{scenario.scenario}-{replicate}-data",
        "train_subject_hash": f"{scenario.scenario}-{replicate}-train",
        "test_subject_hash": f"{scenario.scenario}-{replicate}-test",
        "design_id": f"{scenario.scenario}-design",
        "provenance": "fixture",
    }


def _row(scenario, replicate, method, *, status, marker):
    identity = _cohort_identity(scenario, replicate)
    success = status == "success"
    row = {field: "" for field in strict.RESULT_FIELDS}
    row.update(
        schema_version=strict.SCHEMA_VERSION,
        mode="formal",
        phase=scenario.phase,
        example=scenario.example,
        protocol=scenario.protocol,
        scenario=scenario.scenario,
        replicate=replicate,
        seed=identity["seed"],
        split_seed=identity["split_seed"],
        split_unit="subject",
        method=method,
        method_display_name=METHOD_SPECS[method].display_name,
        method_version="fixture/old" if marker == "old" else "fixture/current",
        applicability="applicable" if success else "N/A by design",
        applicability_reason="fixture",
        admission_status="not_required" if method == strict.TRACE else "admitted",
        attempt_status=status,
        converged=success,
        failure_code="" if success else "not_applicable",
        failure_message="",
        design_id=identity["design_id"],
        provenance=identity["provenance"],
        n_subjects=4,
        n_train_subjects=3,
        n_test_subjects=1,
        n_rows=8,
        n_covariates=2,
        n_active=2,
        has_null_blocks=False,
        data_hash=identity["data_hash"],
        train_subject_hash=identity["train_subject_hash"],
        test_subject_hash=identity["test_subject_hash"],
        tuning_json="{\"fixture\":true}",
        tuning_sha256="a" * 64,
        realized_tuning_json="{\"fixture\":true}",
        realized_tuning_sha256="b" * 64,
        runtime_seconds=0.1 if success else float("nan"),
        peak_python_memory_mb=1.0 if success else float("nan"),
        observed_test_mspe=0.1 if success else float("nan"),
        test_mse=0.1 if success else float("nan"),
        noise_free_test_mspe=0.1 if success else float("nan"),
        baseline_ise=0.1 if success else float("nan"),
        component_ise=99.0 if marker == "old" and method == strict.HHY else 0.1,
        factor_ise=99.0 if marker == "old" and method == strict.HHY else 0.1,
        paper_observed_factor_mse_total=0.1 if success else float("nan"),
        paper_training_function_mse_total=0.1 if success else float("nan"),
        tpr=float("nan"),
        fdr=float("nan"),
        model_size=float("nan"),
        selected_blocks_json="[0,1]" if success else "[]",
        fit_metadata_json=json.dumps({"marker": marker}, sort_keys=True),
    )
    return row


def _test_row_ids(scenario, replicate):
    return tuple(f"{scenario.scenario}-test-{replicate}-{index}" for index in range(2))


def _prediction_rows(scenario, replicate, method, *, marker):
    identity = _cohort_identity(scenario, replicate)
    return [
        {
            "schema_version": strict.SCHEMA_VERSION,
            "scenario": scenario.scenario,
            "replicate": replicate,
            "seed": identity["seed"],
            "method": method,
            "row_id": row_id,
            "subject_id": f"subject-{replicate}",
            "observed_response": float(index),
            "noise_free_target": float(index),
            "prediction": float(index) + (1.0 if marker == "old" else 0.25),
        }
        for index, row_id in enumerate(_test_row_ids(scenario, replicate))
    ]


def _curve(scenario, replicate, method, *, marker):
    identity = _cohort_identity(scenario, replicate)
    return {
        "schema_version": strict.SCHEMA_VERSION,
        "scenario": scenario.scenario,
        "replicate": replicate,
        "seed": identity["seed"],
        "method": method,
        "curves": [
            {
                "component": "baseline",
                "domain": "time",
                "grid": [0.0, 1.0],
                "values": [9.0 if marker == "old" and method == strict.HHY else 0.25, 0.0],
            }
        ],
    }


def _source_success_methods(scenario):
    if scenario.example == "Example 1":
        return {strict.TRACE, strict.ZW}
    if scenario.example == "Example 2":
        return {strict.TRACE, strict.ZZW, strict.HHY, strict.ZY}
    raise AssertionError("Example 3 must not be in the fixture source prefix")


def _write_source_prefix(
    root: Path,
    *,
    example2_replications=2,
    jobs=1,
    retain_failed_zy_artifacts=False,
):
    source = root / "source"
    source.mkdir()
    tasks = _tasks(example2_replications=example2_replications)
    # The last registered task is Example 3, so the preceding complete
    # Example-1/2 sequence is exactly the repair scope.
    prefix_tasks = tasks[:-1]
    result_rows = []
    prediction_rows = []
    curve_rows = []
    for scenario, replicate in prefix_tasks:
        successes = _source_success_methods(scenario)
        for method in FIXED_METHOD_LABELS:
            status = "success" if method in successes else "N/A by design"
            retained_failed_artifact = (
                retain_failed_zy_artifacts
                and scenario.example == "Example 2"
                and replicate == 1
                and method == strict.ZY
            )
            if retained_failed_artifact:
                status = "failed"
            result_rows.append(_row(scenario, replicate, method, status=status, marker="old"))
            if (status == "success" or retained_failed_artifact) and method != strict.ZSY:
                prediction_rows.extend(_prediction_rows(scenario, replicate, method, marker="old"))
            if status == "success" or retained_failed_artifact:
                curve_rows.append(_curve(scenario, replicate, method, marker="old"))
    results = source / "strict_results.csv"
    predictions = source / "strict_predictions.csv"
    curves = source / "strict_factor_curves.jsonl"
    results.write_bytes(strict._csv_bytes(strict.RESULT_FIELDS, result_rows, include_header=True))
    predictions.write_bytes(
        strict._csv_bytes(strict.PREDICTION_FIELDS, prediction_rows, include_header=True)
    )
    curves.write_bytes(strict._jsonl_bytes(curve_rows))
    contract = _contract(jobs=jobs)
    last_scenario, last_replicate = prefix_tasks[-1]
    progress = {
        "schema_version": strict.PROGRESS_SCHEMA_VERSION,
        "run_fingerprint": strict._run_fingerprint(contract),
        "run_contract": contract,
        "expected_cohorts": len(tasks),
        "committed_cohorts": len(prefix_tasks),
        "committed_offsets": {
            "results": results.stat().st_size,
            "predictions": predictions.stat().st_size,
            "curves": curves.stat().st_size,
        },
        "committed_sha256": {
            "results": _hash(results),
            "predictions": _hash(predictions),
            "curves": _hash(curves),
        },
        "last_completed": {
            "phase": last_scenario.phase,
            "scenario": last_scenario.scenario,
            "replicate": last_replicate,
        },
        "status": "running",
    }
    (source / "strict_progress.json").write_text(
        json.dumps(progress, sort_keys=True), encoding="utf-8"
    )
    migration = {
        "schema_version": "vcam-strict-prefix-migration/1",
        "source": {"source_sha256": {"benchmarks/adapters/hhy2021.py": "old-hhy"}},
        "destination": {"source_sha256": contract["source_sha256"]},
    }
    (source / "strict_prefix_migration.json").write_text(
        json.dumps(migration, sort_keys=True), encoding="utf-8"
    )
    return source, contract, tasks


class _HHYAdapter:
    label = strict.HHY

    def preflight(self):
        return PreflightReport(True, "fixture-hhy/2", environment={"fixture": "current"})


def _fake_rebuild(scenario, replicate, root_seed, *, wrong_data_hash=False):
    del root_seed
    identity = _cohort_identity(scenario, replicate)
    data_hash = identity["data_hash"] + ("-wrong" if wrong_data_hash else "")
    raw = SimpleNamespace(
        design_id=identity["design_id"], provenance=identity["provenance"]
    )
    dataset = SimpleNamespace(data_hash=data_hash)
    test = SimpleNamespace(row_id=_test_row_ids(scenario, replicate))
    split = SimpleNamespace(
        train_hash=identity["train_subject_hash"], test_hash=identity["test_subject_hash"]
    )
    return (
        raw,
        dataset,
        SimpleNamespace(),
        test,
        split,
        identity["seed"],
        identity["split_seed"],
    )


class HHYLowdimRepairTests(unittest.TestCase):
    def _patches(self, contract, tasks, calls, *, wrong_data_hash=False):
        def rerun(adapter, scenario, raw, dataset, train, test, split, **kwargs):
            del adapter, raw, dataset, train, split
            replicate = kwargs["replicate"]
            calls.append((scenario.scenario, replicate, kwargs["mode"], kwargs["quick"]))
            row = _row(scenario, replicate, strict.HHY, status="success", marker="new")
            return (
                row,
                _prediction_rows(scenario, replicate, strict.HHY, marker="new"),
                _curve(scenario, replicate, strict.HHY, marker="new"),
            )

        return (
            patch.object(repair, "_current_contract", return_value=(contract, tasks)),
            patch.object(repair, "_rebuild_cohort", side_effect=lambda *args: _fake_rebuild(*args, wrong_data_hash=wrong_data_hash)),
            patch.object(strict, "adapter_registry", return_value={strict.HHY: _HHYAdapter()}),
            patch.object(strict, "_safe_applicability", return_value=("applicable", "fixture")),
            patch.object(strict, "run_one_method", side_effect=rerun),
        )

    def test_replaces_only_example2_hhy_and_audits_every_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, contract, tasks = _write_source_prefix(root)
            before = {
                name: _hash(source / name)
                for name in (
                    "strict_results.csv",
                    "strict_predictions.csv",
                    "strict_factor_curves.jsonl",
                    "strict_progress.json",
                )
            }
            output = root / "repaired"
            calls = []
            patches = self._patches(contract, tasks, calls)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                observed = repair.repair(
                    repair.parse_args(
                        [
                            "--source",
                            str(source),
                            "--output",
                            str(output),
                            "--jobs",
                            "1",
                            "--prefix-cohorts",
                            "3",
                        ]
                    )
                )
            self.assertEqual(observed, output)
            self.assertEqual(
                calls,
                [("repair-example2", 0, "formal", False), ("repair-example2", 1, "formal", False)],
            )
            after = {
                name: _hash(source / name)
                for name in (
                    "strict_results.csv",
                    "strict_predictions.csv",
                    "strict_factor_curves.jsonl",
                    "strict_progress.json",
                )
            }
            self.assertEqual(after, before)
            self.assertFalse((output / ".replacement_hhy_predictions.csv").exists())
            self.assertFalse((output / ".replacement_hhy_curves.jsonl").exists())
            self.assertFalse((output / repair.TEMPORARY_MARKER_FILENAME).exists())
            self.assertEqual(
                (output / "source_strict_prefix_migration.json").read_bytes(),
                (source / "strict_prefix_migration.json").read_bytes(),
            )
            with (source / "strict_results.csv").open("r", encoding="utf-8", newline="") as handle:
                old_rows = list(csv.DictReader(handle))
            with (output / "strict_results.csv").open("r", encoding="utf-8", newline="") as handle:
                new_rows = list(csv.DictReader(handle))
            self.assertEqual(len(old_rows), len(new_rows))
            for old, new in zip(old_rows, new_rows, strict=True):
                target = old["example"] == "Example 2" and old["method"] == strict.HHY
                if target:
                    self.assertEqual(json.loads(new["fit_metadata_json"])["marker"], "new")
                    self.assertNotEqual(old["component_ise"], new["component_ise"])
                else:
                    self.assertEqual(old, new)
            progress = json.loads((output / "strict_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["committed_cohorts"], 3)
            self.assertEqual(progress["run_contract"], contract)
            for key, filename in repair.STREAM_FILENAMES.items():
                artifact = output / filename
                self.assertEqual(artifact.stat().st_size, progress["committed_offsets"][key])
                self.assertEqual(_hash(artifact), progress["committed_sha256"][key])
            lineage = json.loads(
                (output / repair.REPAIR_LINEAGE_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(lineage["repair_scope"]["method"], strict.HHY)
            self.assertEqual(lineage["cohort_identity_validation"]["repaired_cohorts"], 2)
            self.assertEqual(lineage["cohort_identity_validation"]["successful_repaired_fits"], 2)
            for stream in strict.OUTPUT_STREAM_KEYS:
                self.assertTrue(lineage["stream_record_audit"][stream]["unmodified"]["verified_equal"])
            self.assertEqual(progress["repair_lineage_sha256"], _hash(output / repair.REPAIR_LINEAGE_FILENAME))

    def test_preserves_non_target_failed_method_artifacts_from_actual_stream(self):
        """A historical failed result may still have a valid emitted payload."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, contract, tasks = _write_source_prefix(
                root, retain_failed_zy_artifacts=True
            )
            output = root / "repaired"
            calls = []
            patches = self._patches(contract, tasks, calls)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                repair.repair(
                    repair.parse_args(
                        [
                            "--source",
                            str(source),
                            "--output",
                            str(output),
                            "--jobs",
                            "1",
                            "--prefix-cohorts",
                            "3",
                        ]
                    )
                )

            target = {
                "scenario": "repair-example2",
                "replicate": "1",
                "method": strict.ZY,
            }
            with (source / "strict_predictions.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                source_predictions = [
                    row
                    for row in csv.DictReader(handle)
                    if all(row[field] == value for field, value in target.items())
                ]
            with (output / "strict_predictions.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                output_predictions = [
                    row
                    for row in csv.DictReader(handle)
                    if all(row[field] == value for field, value in target.items())
                ]
            self.assertTrue(source_predictions)
            self.assertEqual(output_predictions, source_predictions)

            def failed_zy_curves(path):
                records = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line:
                        continue
                    record = json.loads(line)
                    if (
                        record["scenario"] == target["scenario"]
                        and str(record["replicate"]) == target["replicate"]
                        and record["method"] == target["method"]
                    ):
                        records.append(record)
                return records

            self.assertEqual(
                failed_zy_curves(output / "strict_factor_curves.jsonl"),
                failed_zy_curves(source / "strict_factor_curves.jsonl"),
            )
            with (output / "strict_results.csv").open("r", encoding="utf-8", newline="") as handle:
                output_results = list(csv.DictReader(handle))
            retained_result = next(
                row
                for row in output_results
                if all(row[field] == value for field, value in target.items())
            )
            self.assertEqual(retained_result["attempt_status"], "failed")

    def test_postcompute_merge_failure_preserves_complete_hhy_spools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, contract, tasks = _write_source_prefix(root)
            output = root / "repaired"
            temporary = output.with_name(output.name + ".hhy-repairing")
            calls = []
            patches = self._patches(contract, tasks, calls)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patch.object(
                    repair,
                    "_rewrite_predictions",
                    side_effect=RuntimeError("forced prediction merge failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced prediction merge failure"):
                    repair.repair(
                        repair.parse_args(
                            [
                                "--source",
                                str(source),
                                "--output",
                                str(output),
                                "--jobs",
                                "1",
                                "--prefix-cohorts",
                                "3",
                            ]
                        )
                    )
            self.assertFalse(output.exists())
            self.assertTrue(temporary.is_dir())
            self.assertGreater(
                (temporary / ".replacement_hhy_predictions.csv").stat().st_size, 0
            )
            self.assertGreater(
                (temporary / ".replacement_hhy_curves.jsonl").stat().st_size, 0
            )
            marker = json.loads(
                (temporary / repair.TEMPORARY_MARKER_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(marker["status"], "postcompute_failed_preserved")
            self.assertEqual(marker["stage"], "replacement_spools_complete")
            self.assertEqual(marker["failure"]["type"], "RuntimeError")
            self.assertIn("forced prediction merge failure", marker["failure"]["message"])
            self.assertEqual(
                calls,
                [("repair-example2", 0, "formal", False), ("repair-example2", 1, "formal", False)],
            )

    def test_rejects_source_progress_hash_mismatch_before_any_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, contract, tasks = _write_source_prefix(root)
            result_path = source / "strict_results.csv"
            payload = result_path.read_bytes()
            result_path.write_bytes(payload.replace(b"fixture/old", b"fixture/bad", 1))
            calls = []
            patches = self._patches(contract, tasks, calls)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(RuntimeError, "prefix hash mismatch"):
                    repair.repair(
                        repair.parse_args(
                            [
                                "--source",
                                str(source),
                                "--output",
                                str(root / "bad"),
                                "--jobs",
                                "1",
                                "--prefix-cohorts",
                                "3",
                            ]
                        )
                    )
            self.assertEqual(calls, [])

    def test_rejects_regenerated_data_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, contract, tasks = _write_source_prefix(root)
            calls = []
            patches = self._patches(contract, tasks, calls, wrong_data_hash=True)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(RuntimeError, "DGP/split identity mismatch"):
                    repair.repair(
                        repair.parse_args(
                            [
                                "--source",
                                str(source),
                                "--output",
                                str(root / "bad"),
                                "--jobs",
                                "1",
                                "--prefix-cohorts",
                                "3",
                            ]
                        )
                    )
            # Identity is checked by the parent immediately after the first
            # ordered worker result; no output is published despite that one
            # deterministic worker evaluation.
            self.assertEqual(calls, [("repair-example2", 0, "formal", False)])
            self.assertFalse((root / "bad").exists())

    def test_parallel_hhy_repair_uses_bounded_ordered_scheduler(self):
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
                self.maximum_outstanding = max(self.maximum_outstanding, self.outstanding)
                return FakeFuture(self, function, args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = 2
            source, contract, tasks = _write_source_prefix(
                root, example2_replications=8, jobs=jobs
            )
            output = root / "repaired"
            calls = []
            patches = self._patches(contract, tasks, calls)
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patch.object(repair, "_configure_worker_threads"),
                patch.object(strict.concurrent.futures, "ProcessPoolExecutor", FakeExecutor),
            ):
                repair.repair(
                    repair.parse_args(
                        [
                            "--source",
                            str(source),
                            "--output",
                            str(output),
                            "--jobs",
                            str(jobs),
                            "--prefix-cohorts",
                            str(len(tasks) - 1),
                        ]
                    )
                )
            executor = FakeExecutor.instance
            self.assertIsNotNone(executor)
            self.assertEqual(executor.max_workers, jobs)
            self.assertEqual(
                executor.maximum_outstanding, strict._ordered_task_prefetch_limit(jobs)
            )
            self.assertLessEqual(
                executor.maximum_outstanding,
                strict.ORDERED_TASK_PREFETCH_FACTOR * jobs,
            )
            self.assertEqual(executor.outstanding, 0)
            self.assertEqual(
                calls,
                [("repair-example2", replicate, "formal", False) for replicate in range(8)],
            )
            lineage = json.loads(
                (output / repair.REPAIR_LINEAGE_FILENAME).read_text(encoding="utf-8")
            )
            execution = lineage["cohort_identity_validation"]["execution"]
            self.assertEqual(execution["jobs"], jobs)
            self.assertEqual(execution["submitted_hhy_cohorts"], 8)
            self.assertEqual(
                execution["max_outstanding_futures"], strict._ordered_task_prefetch_limit(jobs)
            )
            self.assertTrue(execution["canonical_yield_order"])

    def test_stale_temporary_requires_explicit_cleanup_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, contract, tasks = _write_source_prefix(root)
            output = root / "repaired"
            temporary = output.with_name(output.name + ".hhy-repairing")
            temporary.mkdir()
            sentinel = temporary / "interrupted-artifact.txt"
            sentinel.write_text("interrupted", encoding="utf-8")
            calls = []
            patches = self._patches(contract, tasks, calls)
            base_argv = [
                "--source",
                str(source),
                "--output",
                str(output),
                "--jobs",
                "1",
                "--prefix-cohorts",
                "3",
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(RuntimeError, "temporary sibling already exists"):
                    repair.repair(repair.parse_args(base_argv))
                self.assertTrue(sentinel.exists())
                repair.repair(repair.parse_args([*base_argv, "--discard-stale-temporary"]))
            self.assertTrue(output.is_dir())
            self.assertFalse(temporary.exists())
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
