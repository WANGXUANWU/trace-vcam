"""Hash verification for vendored author code; never modifies vendor files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "zsy2026_vcampackage"
ORIGIN_PATH = VENDOR_ROOT / "ORIGIN.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_zsy2026_vendor() -> dict[str, object]:
    with ORIGIN_PATH.open("r", encoding="utf-8") as handle:
        origin = json.load(handle)
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for relative, expected in origin["files"].items():
        path = VENDOR_ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        observed = sha256_file(path)
        if observed != expected:
            mismatched.append(
                {"path": relative, "expected": expected, "observed": observed}
            )
    return {
        "valid": not missing and not mismatched,
        "commit": origin["commit"],
        "package_version": origin["package_version"],
        "missing": missing,
        "mismatched": mismatched,
        "origin": origin,
    }
