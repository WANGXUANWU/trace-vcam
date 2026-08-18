"""Full-data MACS fits of every applicable method, on one identified scale.

The cross-validation run stores prediction errors for all methods but only one
set of factor curves, because the fold curves of a multiplicative model cannot
be averaged.  The application figure that compares estimated shapes therefore
needs one full-data fit per method.  This script produces exactly that: each
method is fitted once on all primary subjects with its own registered tuning
rule, and its baseline and factor curves are written after the same Lebesgue
identification used for the proposed estimator.
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

from scripts.run_macs_application import (  # noqa: E402
    DEFAULT_SEED,
    _macs_applicability,
    _tuning,
    adapter_registry,
    prepare_macs_variant,
    read_macs_csv,
)
from scripts.run_macs_bootstrap import identified_curves  # noqa: E402

GRID_SIZE = 401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "macs_method_curves"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv")
    dataset = prepare_macs_variant(raw, variant="primary")
    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    registry = adapter_registry()

    payload: dict[str, object] = {
        "schema_version": "vcam-macs-method-curves/1",
        "fit_scope": "full primary data; all subjects and observations",
        "seed": int(args.seed),
        "basis_dimension": int(args.basis),
        "n_subjects": int(dataset.n_subjects),
        "n_rows": int(dataset.n_rows),
        "data_hash": str(dataset.data_hash),
        "grid": grid.tolist(),
        "methods": {},
    }

    for method, adapter in registry.items():
        applicability, reason = _macs_applicability(method)
        entry: dict[str, object] = {
            "applicability": applicability,
            "applicability_reason": reason,
        }
        if applicability != "applicable":
            payload["methods"][method] = entry
            print(f"{method}: {applicability} ({reason})")
            continue
        preflight = adapter.preflight()
        if not preflight.ready:
            entry["attempt_status"] = "failed"
            entry["failure_code"] = str(preflight.code)
            entry["failure_message"] = str(preflight.message)[:500]
            payload["methods"][method] = entry
            print(f"{method}: preflight not ready ({preflight.code})")
            continue
        tuning = _tuning(method, args.basis, quick=False)
        started = time.perf_counter()
        try:
            artifact = adapter.fit(dataset, seed=args.seed, tuning=tuning)
            curves = identified_curves(adapter, artifact, grid)
        except Exception as error:
            entry["attempt_status"] = "failed"
            entry["failure_code"] = str(getattr(error, "code", type(error).__name__))
            entry["failure_message"] = f"{type(error).__name__}: {error}"[:500]
            payload["methods"][method] = entry
            print(f"{method}: FAILED {entry['failure_code']}")
            continue
        entry["attempt_status"] = "success" if artifact.converged else "failed"
        entry["converged"] = bool(artifact.converged)
        entry["runtime_seconds"] = float(time.perf_counter() - started)
        entry["curves"] = {
            name: np.asarray(values, dtype=float).tolist()
            for name, values in curves.items()
        }
        payload["methods"][method] = entry
        print(
            f"{method}: {entry['attempt_status']} in "
            f"{entry['runtime_seconds']:.1f}s"
        )

    target = args.output / "macs_method_curves.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
