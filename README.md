# TRACE-VCAM

Reference implementation and reproduction code for

> **Robust Low-Rank Estimation of Varying-Coefficient Additive Models with Longitudinal Covariates**

**TRACE-VCAM** stands for *trace-norm regularised convex estimation of a varying-coefficient additive model*. A varying-coefficient additive model writes the effect of covariate `k` as a product of a time-varying coefficient and a univariate additive function,

```
E[Y_ij | T_ij, Z_ij] = beta_0(T_ij) + sum_k beta_k(T_ij) * phi_k(Z_ijk),
```

which is interpretable but nonconvex to fit directly. After centred and whitened marginal spline transformations, a separable component has a **rank-one** coefficient matrix, so separability becomes a rank restriction rather than an alternating search. The estimator has two stages:

1. **Convex pilot.** One convex program combining a subject-balanced Huber loss, a block trace norm (nuclear norm) that can delete a whole covariate, and a normalised tensor roughness penalty.
2. **Rank-one projection and scalar postfit.** Each retained block is projected onto its leading singular component, the directions are then held fixed, and only the baseline and one amplitude per block are refitted by a low-dimensional convex Huber regression.

Because the pilot is convex, the matrix that the projection acts on is a global minimiser, so factor recovery follows from singular-vector perturbation rather than from the limit point of a backfitting loop.

---

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/WANGXUANWU/trace-vcam.git
cd trace-vcam
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The two R-based competitors additionally require R with the packages pinned in `environment/r-packages.lock.txt`.

## Quick start

Run from the repository root, so that `src` is importable as a package path.

```python
from src.trace_vcam import (
    OrthonormalSplineBasis,
    VCAMDesign,
    fit_trace_vcam,
    practical_huber_threshold,
    recover_factors,
    trace_lambda_max,
)

# t: (N,) times mapped to [0, 1]        z: (N, p) covariates mapped to [0, 1]
# y: (N,) responses                     subject: (N,) integer subject labels
basis = OrthonormalSplineBasis.create(time_dimension=6, covariate_dimension=6)
design = VCAMDesign.from_arrays(
    time=t, covariates=z, response=y, subject=subject, basis=basis
)

delta, scale = practical_huber_threshold(design)      # tau = 1.345 * MAD scale
lambda_max, _ = trace_lambda_max(design, delta)
fit = fit_trace_vcam(design, penalty=0.03 * lambda_max, delta=delta, mu=0.05)

fit.selected              # boolean mask: the blocks the trace norm retains
fit.predict(design)       # fitted mean, from the scalar postfit
factors = recover_factors(fit.matrices[0], basis)     # (beta_1, phi_1) coefficients
```

`VCAMDesign.from_arrays` gives every subject total weight `1/n`, so a heavily observed subject does not dominate the fit and cluster size may be informative. `recover_factors` applies the identification used in the paper: the time factor integrates to one and the covariate factor integrates to zero over the mapped unit interval, which fixes the scale and sign of the pair without changing the fitted surface. It returns `None` for a block whose time factor has a numerically zero integral.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/trace_vcam.py` | The two-stage estimator: convex pilot, rank-one projection, scalar postfit, factor normalisation. |
| `src/trace_vcam_roughness.py` | Operator-normalised tensor curvature matrices. |
| `src/trace_tuning_protocol.py` | The locked tuning protocol and the `lambda_max` definition. |
| `src/simulation_dgp.py`, `experiments/dgp.py` | Registered data-generating mechanisms for Examples 1--3 and for the published source designs. |
| `benchmarks/` | Audited adapters for the published comparison methods, the applicability rules, and the failure policy. |
| `protocol/` | Frozen tuning locks and the registered published targets, as JSON. |
| `scripts/` | Runners for the simulations and the application, and the table and figure builders. |
| `tests/` | Unit and protocol tests, including the reproduction audits. |
| `data/` | Provenance of the MACS CD4 data, with the CSV export and its checksums. |
| `environment/` | Pinned R package versions for the R-based competitors. |

## Reproducing the paper

Each runner writes to `results/` and each builder reads only from an accepted result file, so a table can never be produced from an unaudited run.

```bash
# Simulations: Examples 1 and 2, all registered sample sizes and error laws
python scripts/run_strict_benchmark.py

# Example 3: block-sparse design with p = 6 and two active blocks
python scripts/run_robust_example.py

# The two supplementary checks on the theoretical conditions (Section S.4)
python scripts/run_theory_gap_checks.py

# Application: repeated subject-level cross-validation on the MACS CD4 data
python scripts/run_macs_application.py
python scripts/run_macs_bootstrap.py
python scripts/run_macs_stability.py

# Tables and figures
python scripts/analyze_strict_results.py
python scripts/build_manuscript_outputs.py
python scripts/build_macs_outputs.py
```

Run the tests with

```bash
python -m pytest tests -q
```

## Benchmark protocol

The comparison admits a method only if a runnable original, author, or explicitly paper-aligned implementation exists, and only inside the sampling regime for which that method was derived. A method outside its stated regime is reported as **not applicable** rather than modified until it runs, and every attempted fit stays in the denominator of the reported failure counts. Within a replication all applicable methods receive the same generated data, the same seed, and the same subject-level training and test split, and splits are never made inside a subject. Section S.5 of the Supplementary Material states the full contract.

## Third-party code

The comparison uses the authors' own implementation wherever one is distributed. Those sources are **not** redistributed here.

- Zhang and Wang (2015) is run from the CRAN package `fdapace`.
- Zhao, Sun and Yang (2026) is run from the authors' repository, pinned at commit `27d857a71807de807761a022a4e334745737761e`. `benchmarks/vendor.py` fetches and verifies it; a source hash mismatch stops a run before any fit is attempted.
- Zhao and Yang (2025) distributes no software, so `benchmarks/adapters/zy2025.py` is an independent implementation of the published algorithm. It is labelled as such everywhere and is not attributed to the authors.

## Data

`data/raw/catdata_aids.csv` is a lossless CSV export of the `aids` object in CRAN `catdata` 1.2.5, covering 2,376 records on 369 men who became HIV positive during follow-up. `data/README.md` records the canonical source, the checksums, and the coordinate mapping used by the analysis.

## Citation

```bibtex
@article{tracevcam,
  title  = {Robust Low-Rank Estimation of Varying-Coefficient Additive Models
            with Longitudinal Covariates},
  author = {Anonymous},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). The pinned third-party sources retain the licences of their own authors.
