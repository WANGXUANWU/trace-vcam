import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_overleaf_clean import (
    FIGURE_FILES,
    WHITELIST,
    StagingError,
    manifest_path_for,
    prepare,
    sha256_file,
)


def _make_complete_source(root: Path, *, artifacts_ready: bool = True) -> Path:
    source = root / "source"
    for relative in WHITELIST:
        path = source / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in FIGURE_FILES:
            path.write_bytes(b"%PDF-1.4\n% fixture vector figure\n%%EOF\n")
        elif relative == "tables/strict_claims.tex":
            switch = (
                r"\strictartifactsreadytrue"
                if artifacts_ready
                else r"\strictartifactsreadyfalse"
            )
            path.write_text(switch + "\n", encoding="utf-8")
        else:
            path.write_text(f"% fixture: {relative}\n", encoding="utf-8")
    return source


class CleanOverleafStagingTests(unittest.TestCase):
    def test_success_is_exactly_whitelist_and_manifest_is_external(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_complete_source(root)
            output = root / "VCAM_overleaf"
            manifest = prepare(source, output)

            observed = {
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(observed, set(WHITELIST))
            self.assertEqual(len(observed), 40)
            external = manifest_path_for(output)
            self.assertTrue(external.is_file())
            self.assertFalse((output / external.name).exists())
            self.assertEqual(json.loads(external.read_text(encoding="utf-8")), manifest)

    def test_missing_formal_artifact_or_false_switch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_complete_source(root, artifacts_ready=False)
            (source / FIGURE_FILES[0]).unlink()
            with self.assertRaisesRegex(StagingError, "Missing formal artifact"):
                prepare(source, root / "staging")

            (source / FIGURE_FILES[0]).write_bytes(b"%PDF-1.4\n%%EOF\n")
            with self.assertRaisesRegex(StagingError, "disables strict artifact"):
                prepare(source, root / "staging")

    def test_forbidden_source_extras_are_not_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_complete_source(root)
            extras = (
                "main.pdf",
                "build/main.aux",
                "raw/results.csv",
                "analysis.json",
                "code/run.py",
                "tables__legacy.tex",
                "figures__legacy.pdf",
            )
            for relative in extras:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("forbidden extra\n", encoding="utf-8")

            output = root / "staging"
            prepare(source, output)
            for relative in extras:
                self.assertFalse((output / relative).exists())
            self.assertEqual(
                {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()},
                set(WHITELIST),
            )

    def test_manifest_hashes_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_complete_source(root)
            output = root / "staging"
            preview = prepare(source, output, dry_run=True)
            self.assertEqual(preview["status"], "validated-dry-run")
            self.assertFalse(output.exists())
            self.assertFalse(manifest_path_for(output).exists())

            manifest = prepare(source, output)
            self.assertEqual(manifest["package_file_count"], 40)
            entries = manifest["files"]
            self.assertEqual([entry["path"] for entry in entries], list(WHITELIST))
            self.assertIn("tables/extreme_finite_audit.tex", WHITELIST)
            for entry in entries:
                staged = output / entry["path"]
                self.assertEqual(entry["sha256"], sha256_file(staged))
                self.assertEqual(entry["source_sha256"], entry["sha256"])
                self.assertEqual(entry["bytes"], staged.stat().st_size)

            digest = hashlib.sha256()
            for entry in entries:
                digest.update(entry["path"].encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(entry["bytes"]).encode("ascii"))
                digest.update(b"\0")
                digest.update(entry["sha256"].encode("ascii"))
                digest.update(b"\n")
            self.assertEqual(manifest["package_sha256"], digest.hexdigest())


if __name__ == "__main__":
    unittest.main()
