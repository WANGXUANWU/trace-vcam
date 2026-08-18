"""Create the final, whitelist-only Overleaf staging directory.

This tool intentionally knows nothing about simulation runners or raw result
directories.  It accepts only the 40 files in the audited LaTeX dependency
closure and refuses to stage the manuscript until every generated table and
figure exists and ``strict_claims.tex`` enables the artifact-ready switch.

The audit manifest is written next to the staging directory, never inside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable


WORKSPACE = Path(__file__).resolve().parents[1]

ROOT_FILES = (
    "main.tex",
    "supplement.tex",
    "references.bib",
)

SECTION_FILES = (
    "sections/00_abstract.tex",
    "sections/01_introduction.tex",
    "sections/02_model_estimator.tex",
    "sections/03_theory.tex",
    "sections/04_simulations.tex",
    "sections/05_macs.tex",
    "sections/06_discussion.tex",
    "sections/citation-style.tex",
    "sections/preamble.tex",
)

SUPPLEMENT_FILES = (
    "supplement/01_computation.tex",
    "supplement/02_assumptions.tex",
    "supplement/03_proofs.tex",
    "supplement/04_benchmark_audit.tex",
    "supplement/05_additional_results.tex",
    "supplement/06_reproducibility.tex",
)

TABLE_FILES = (
    "tables/strict_claims.tex",
    "tables/example1_main.tex",
    "tables/example2_main.tex",
    "tables/example3_main.tex",
    "tables/scaling_main.tex",
    "tables/macs_cv_main.tex",
    "tables/method_admission.tex",
    "tables/example1_full.tex",
    "tables/example2_full.tex",
    "tables/example3_full.tex",
    "tables/scaling_full.tex",
    "tables/failure_audit.tex",
    "tables/extreme_finite_audit.tex",
    "tables/macs_sensitivity.tex",
    "tables/result_manifest.tex",
)

FIGURE_FILES = (
    "figures/example1_factor_recovery.pdf",
    "figures/example2_robustness.pdf",
    "figures/macs_components.pdf",
    "figures/macs_surfaces.pdf",
    "figures/supp_example1_components.pdf",
    "figures/supp_example2_distributions.pdf",
    "figures/supp_example3_selection.pdf",
)

WHITELIST = ROOT_FILES + SECTION_FILES + SUPPLEMENT_FILES + TABLE_FILES + FIGURE_FILES
FORMAL_ARTIFACTS = TABLE_FILES + FIGURE_FILES

if len(WHITELIST) != 40 or len(set(WHITELIST)) != 40:  # pragma: no cover
    raise RuntimeError("The audited Overleaf whitelist must contain exactly 40 paths")


class StagingError(RuntimeError):
    """Raised when the manuscript is not safe or complete enough to stage."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path_for(output: Path) -> Path:
    """Return the package-external manifest path for ``output``."""

    return output.parent / f"{output.name}.manifest.json"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_source(source: Path) -> Path:
    try:
        resolved = source.expanduser().resolve(strict=True)
    except OSError as exc:
        raise StagingError(f"Source directory is unavailable: {source} ({exc})") from exc
    if not resolved.is_dir():
        raise StagingError(f"Source is not a directory: {resolved}")
    return resolved


def _resolve_output(output: Path, source: Path) -> Path:
    resolved = output.expanduser().resolve(strict=False)
    if resolved == resolved.parent:
        raise StagingError("The filesystem root cannot be used as a staging directory")
    if resolved == source or _is_relative_to(resolved, source):
        raise StagingError("The staging directory must be outside the manuscript source")
    if _is_relative_to(source, resolved):
        raise StagingError("The staging directory cannot contain the manuscript source")
    if resolved.exists() and resolved.is_symlink():
        raise StagingError(f"Refusing a symlink staging directory: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise StagingError(f"Staging path is not a directory: {resolved}")
    return resolved


def _strip_tex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line if match is None else line[: match.start()])
    return "\n".join(lines)


def _validate_artifact_switch(source: Path, errors: list[str]) -> None:
    claims = source / "tables" / "strict_claims.tex"
    if not claims.is_file():
        return
    try:
        active_text = _strip_tex_comments(claims.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"tables/strict_claims.tex is unreadable UTF-8 ({exc})")
        return
    if re.search(r"\\strictartifactsreadyfalse\b", active_text):
        errors.append("tables/strict_claims.tex disables strict artifact readiness")
    if not re.search(r"\\strictartifactsreadytrue\b", active_text):
        errors.append(
            "tables/strict_claims.tex does not enable \\strictartifactsreadytrue"
        )


def validate_source(source: Path) -> None:
    """Validate all 40 inputs and the formal-artifact readiness switch."""

    errors: list[str] = []
    formal = set(FORMAL_ARTIFACTS)
    figures = set(FIGURE_FILES)
    for relative in WHITELIST:
        path = source / Path(relative)
        if not path.is_file():
            kind = "formal artifact" if relative in formal else "manuscript file"
            errors.append(f"Missing {kind}: {relative}")
            continue
        if path.is_symlink():
            errors.append(f"Whitelist input must not be a symlink: {relative}")
            continue
        try:
            if path.stat().st_size <= 0:
                errors.append(f"Whitelist input is empty: {relative}")
                continue
            if relative in figures:
                with path.open("rb") as handle:
                    if handle.read(5) != b"%PDF-":
                        errors.append(f"Formal figure is not a PDF file: {relative}")
        except OSError as exc:
            errors.append(f"Whitelist input is unreadable: {relative} ({exc})")

    _validate_artifact_switch(source, errors)
    if errors:
        rendered = "\n  - ".join(errors)
        raise StagingError(f"Overleaf staging preflight failed:\n  - {rendered}")


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def validate_staging(root: Path) -> None:
    """Prove that a staged package contains the whitelist and nothing else."""

    observed = _relative_file_set(root)
    expected = set(WHITELIST)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise StagingError(
            f"Staging content differs from the 40-file whitelist; "
            f"missing={missing}, extra={extra}"
        )

    allowed_directories = {"sections", "supplement", "tables", "figures"}
    figures = set(FIGURE_FILES)
    forbidden_suffixes = {
        ".aux",
        ".log",
        ".out",
        ".bbl",
        ".blg",
        ".toc",
        ".fls",
        ".fdb_latexmk",
        ".synctex",
        ".csv",
        ".json",
        ".py",
        ".r",
        ".rmd",
        ".zip",
    }
    for relative in sorted(observed):
        path = Path(relative)
        if len(path.parts) > 1 and path.parts[0] not in allowed_directories:
            raise StagingError(f"Forbidden package directory: {relative}")
        lower_name = path.name.lower()
        if lower_name.startswith(("tables__", "figures__")):
            raise StagingError(f"Forbidden flat-upload filename: {relative}")
        suffix = path.suffix.lower()
        if suffix in forbidden_suffixes:
            raise StagingError(f"Forbidden package file type: {relative}")
        if suffix == ".pdf" and relative not in figures:
            raise StagingError(f"Compiled or unregistered PDF is forbidden: {relative}")


def _package_digest(entries: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(source: Path, staged: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for relative in WHITELIST:
        path = staged / Path(relative)
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_sha256": sha256_file(source / Path(relative)),
            }
        )
    whitelist_hash = hashlib.sha256(
        ("\n".join(WHITELIST) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "schema": "vcam-overleaf-clean-v1",
        "package_file_count": len(entries),
        "whitelist_sha256": whitelist_hash,
        "package_sha256": _package_digest(entries),
        "files": entries,
    }


def _destination_preflight(output: Path, *, force: bool) -> None:
    manifest = manifest_path_for(output)
    if output.exists() and any(output.iterdir()) and not force:
        raise StagingError(
            f"Staging directory is not empty; pass --force to replace it: {output}"
        )
    if manifest.exists() and not force:
        raise StagingError(
            f"External manifest already exists; pass --force to replace it: {manifest}"
        )
    if manifest.exists() and not manifest.is_file():
        raise StagingError(f"External manifest path is not a file: {manifest}")


def _remove_destination(output: Path) -> None:
    if not output.exists():
        return
    if output.is_symlink() or not output.is_dir():  # defensive; checked earlier
        raise StagingError(f"Refusing to remove unsafe staging target: {output}")
    shutil.rmtree(output)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare(
    source: Path,
    output: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Validate and, unless ``dry_run``, atomically create the clean staging."""

    source = _resolve_source(Path(source))
    output = _resolve_output(Path(output), source)
    validate_source(source)
    _destination_preflight(output, force=force)

    if dry_run:
        return {
            "schema": "vcam-overleaf-clean-v1",
            "status": "validated-dry-run",
            "package_file_count": len(WHITELIST),
            "output": str(output),
            "manifest": str(manifest_path_for(output)),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        for relative in WHITELIST:
            destination = temporary / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / Path(relative), destination)
        validate_staging(temporary)
        manifest = build_manifest(source, temporary)
        if any(
            entry["sha256"] != entry["source_sha256"]
            for entry in manifest["files"]  # type: ignore[index]
        ):
            raise StagingError("A copied file hash differs from its source hash")

        _remove_destination(output)
        temporary.replace(output)
        _write_json_atomic(manifest_path_for(output), manifest)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the audited 40-file VCAM Overleaf staging directory."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=WORKSPACE / "manuscript",
        help="manuscript source directory (default: workspace/manuscript)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing staging and manifest"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare(
            args.source,
            args.output,
            dry_run=args.dry_run,
            force=args.force,
        )
    except StagingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
