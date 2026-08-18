"""Regression coverage for the VCAM-Lasso non-ASCII-path boundary."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.adapters.external import (
    RUNNER_ROOT,
    ZSY_ASCII_STAGE_ROOT_ENV,
    _r_runtime,
    _stage_zsy2026_author_code,
)
from benchmarks.vendor import VENDOR_ROOT, sha256_file, verify_zsy2026_vendor


class ZSY2026AsciiStageTests(unittest.TestCase):
    def test_windows_r_child_drops_unsupported_inherited_c_utf8(self) -> None:
        """The source wrapper receives a usable UTF-8 R locale on Windows."""

        with patch.dict(
            os.environ,
            {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "LC_CTYPE": "C.UTF-8"},
            clear=False,
        ):
            _, environment = _r_runtime()
        if os.name == "nt":
            self.assertNotIn("LANG", environment)
            self.assertNotIn("LC_ALL", environment)
            self.assertNotIn("LC_CTYPE", environment)

    def test_prepares_verified_ascii_copy_without_running_r(self) -> None:
        """The wrapper and entire pinned snapshot can be sourced from ASCII paths.

        This deliberately exercises preparation only: it must not call R or
        attempt a potentially long VCAM-Lasso fit.
        """

        vendor = verify_zsy2026_vendor()
        self.assertTrue(vendor["valid"])
        with tempfile.TemporaryDirectory(prefix="zsy-ascii-stage-test-") as temporary:
            parent = Path(temporary)
            self.assertTrue(str(parent).isascii())
            with patch.dict(
                os.environ,
                {ZSY_ASCII_STAGE_ROOT_ENV: str(parent)},
                clear=False,
            ):
                with _stage_zsy2026_author_code() as stage:
                    self.assertTrue(str(stage.root).isascii())
                    self.assertTrue(str(stage.runner_path).isascii())
                    self.assertTrue(str(stage.vendor_source_path).isascii())
                    self.assertNotEqual(stage.runner_path, RUNNER_ROOT / "zsy2026_author_code.R")
                    self.assertNotEqual(stage.vendor_source_path, VENDOR_ROOT / "R" / "VCAMLasso.R")
                    self.assertTrue(stage.runner_path.is_file())
                    self.assertTrue(stage.vendor_source_path.is_file())

                    expected_keys = {"runner/zsy2026_author_code.R"}
                    expected_keys.update(
                        "vendor/" + str(relative).replace("\\", "/")
                        for relative in vendor["origin"]["files"]
                    )
                    self.assertEqual(set(stage.source_sha256), expected_keys)
                    self.assertEqual(stage.source_sha256, stage.staged_sha256)
                    self.assertEqual(
                        stage.staged_sha256["runner/zsy2026_author_code.R"],
                        sha256_file(RUNNER_ROOT / "zsy2026_author_code.R"),
                    )
                    self.assertEqual(
                        stage.staged_sha256["vendor/R/VCAMLasso.R"],
                        vendor["origin"]["files"]["R/VCAMLasso.R"],
                    )
                    metadata = stage.audit_metadata()
                    self.assertEqual(
                        metadata["author_code_path_strategy"],
                        "verified-ascii-temporary-staging",
                    )
                    self.assertTrue(metadata["author_code_staged_paths_ascii"])
                    self.assertTrue(metadata["author_code_hashes_match_after_staging"])
                    staged_root = stage.root

            self.assertFalse(staged_root.exists())
        self.assertEqual(
            sha256_file(VENDOR_ROOT / "R" / "VCAMLasso.R"),
            vendor["origin"]["files"]["R/VCAMLasso.R"],
        )


if __name__ == "__main__":
    unittest.main()
