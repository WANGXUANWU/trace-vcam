# Strict VCAM benchmark contract

This directory is the only supported entry point for literature benchmarks.
It does not reuse the retired schema-12 proxy baselines.

## Fixed method identities

- `TRACE-VCAM`: local two-stage estimator.
- `ZW2015`: unmodified CRAN `fdapace::VCAM` through an I/O-only R wrapper.
- `ZZW2020`: paper Algorithm 1, with the time-invariant and longitudinal
  initialization regimes, modified inner/outer backfitting, and the published
  `1e-2`/`1e-3` MRS tolerances.
- `HHY2021-Huber`: paper tensor pilot, additive M-step, and
  varying-coefficient M-step with subject-balanced Huber loss at `delta=1.345`.
- `ZSY2026-author-code`: unmodified author snapshot at commit
  `27d857a71807de807761a022a4e334745737761e`.
- `ZY2025-paper-implementation`: implementation from the supplied paper
  equations; no author code or supplement was available, so it is never
  described as author software.

`methods.py` pre-registers design-level applicability. Runtime dependency
failure is a different state: an applicable attempt that fails preflight stays
in the failure-rate denominator. `N/A by design` records are not attempted.

## One replication

1. Construct a `SubjectDataset` with an explicit `subject_id` per row.
2. Create one `SubjectSplit` and reuse it, the data hash, and the seed for every
   method.
3. Call `run_replication(adapter, dataset, split, protocol=...,
   scenario_id=..., replication_id=..., tuning=...)`.
4. Store every returned `BenchmarkResult`, including dependency failures and
   nonconvergence. `audit_replication_cohort` rejects seed, data, split, or
   method-registry mismatches.

The result schema records method/version, applicability, realized tuning,
convergence, failure stage/reason, runtime, row-linked predictions, identified
factor curves, selected blocks, metrics, and environment/provenance metadata.
Python/R exchange bundles contain observation and subject-split CSV files plus
content hashes.

## Optional source-value diagnostics

`assess_reproduction` compares an optional source-paper run with the published
point and its registered Monte Carlo or rounding tolerance. This diagnostic
does not control entry to the common benchmark. Every scientifically
applicable implementation is compared on the same generated data, seed, and
subject split, and failures remain in `n_attempted`. The Zhao--Yang (2025)
Table 1 targets are transcribed in `admission.py`.

Some source limitations deliberately remain visible:

- the pinned ZSY2026 function has no held-out prediction API, so prediction is
  a metric-level `N/A`; no extrapolator is invented;
- `fdapace` versions and their spline-knot constraints are recorded verbatim;
- locked knot counts are labelled as such and are not represented as a rerun of
  a paper's BIC/CV search;
- the three-step M-VCAM IRLS trace distinguishes strict coefficient-change
  convergence from an explicitly registered finite objective-stability stop
  (three relative Huber-objective changes at most `1e-9`); the loss, stages,
  and BIC formula are unchanged, and the termination state is retained in
  every BIC trace;
- final integral normalization of a three-step M-VCAM coefficient inverse-
  rescales its paired additive function, so the fitted component surface used
  for BIC and prediction is unchanged;
- the ZSY paper/code normalization and returned-`MSE` discrepancies are listed
  in `vendor/zsy2026_vcampackage/ORIGIN.json`. This is a difference audit, not
  a silently corrected "paper-aligned" result.

Run the framework tests with:

```text
python -m unittest tests.test_benchmark_framework -v
```
