"""Full-data component curves under the prespecified MACS perturbations.

The application pipeline fits full-data curves for the primary recipe only.
This script repeats that fit under each registered perturbation -- deleting
subjects flagged by the outer-fence rule, winsorising the response, and
changing the marginal basis size -- so that the stability of the estimated
shapes can be displayed.  Each variant uses the same cross-validated tuning
rule as the primary analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.adapters.trace import TraceVCAMAdapter  # noqa: E402
from scripts.run_macs_application import (  # noqa: E402
    DEFAULT_SEED,
    prepare_macs_variant,
    read_macs_csv,
)
from scripts.run_macs_bootstrap import GRID_SIZE, identified_curves  # noqa: E402

VARIANTS = (
    ("primary", 6),
    ("delete_outer_fence_subjects", 6),
    ("winsorize_response_1_99", 6),
    ("basis_5", 5),
    ("basis_8", 8),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "macs_stability"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv")
    adapter = TraceVCAMAdapter()
    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    payload: dict[str, object] = {
        "schema_version": "vcam-macs-stability/1",
        "seed": int(args.seed),
        "grid": grid.tolist(),
        "variants": {},
    }

    for variant, basis in VARIANTS:
        started = time.perf_counter()
        dataset = prepare_macs_variant(raw, variant=variant)
        tuning = {
            "time_domain": [0.0, 1.0],
            "covariate_domains": [[0.0, 1.0], [0.0, 1.0]],
            "q_time": basis,
            "q_covariate": basis,
            "delta_rule": "mad",
            "huber_multiplier": 1.345,
            "lambda_ratio": 0.03,
            "roughness": 0.5,
            "max_iter": 2000,
            "tolerance": 1e-7,
            "selection": "subject_cv",
            "cv_folds": 3,
            "cv_basis_grid": [basis - 1, basis],
            "cv_lambda_ratio_grid": [0.2, 0.6, 0.9],
            "cv_roughness_grid": [0.5],
            "cv_huber_multiplier_grid": [1.345, 3.0, 10.0],
        }
        artifact = adapter.fit(dataset, seed=args.seed, tuning=tuning)
        curves = identified_curves(adapter, artifact, grid)
        payload["variants"][variant] = {
            "n_subjects": int(dataset.n_subjects),
            "n_rows": int(dataset.n_rows),
            "selected_tuning": dict(artifact.tuning["selection_audit"]["selected"]),
            "curves": {name: values.tolist() for name, values in curves.items()},
        }
        print(
            f"{variant}: {dataset.n_subjects} subjects, "
            f"selected {payload['variants'][variant]['selected_tuning']} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    target = args.output / "macs_variant_curves.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
