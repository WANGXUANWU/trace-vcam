"""Safely correct a completed MACS metadata audit without rerunning any fits.

This command is for the historical final-audit CSV type-normalization defect.
It first verifies that the frozen results, predictions, and curve streams match
both the append journal and the original metadata, then updates only the
metadata/hash pair and writes an immutable pre-fix snapshot plus an audit sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_macs_application import refinalize_existing_macs_metadata  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Existing finalized MACS output directory; raw streams remain untouched.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    paths = refinalize_existing_macs_metadata(parse_args(argv).output)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
