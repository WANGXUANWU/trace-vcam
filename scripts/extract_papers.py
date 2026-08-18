from __future__ import annotations

import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "pdfs" / "text"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, object]] = []
    pdf_paths = sorted(
        [
            *ROOT.glob("*.pdf"),
            *(ROOT / "references" / "papers").glob("*.pdf"),
            *(ROOT / "references" / "supplements").glob("*.pdf"),
        ]
    )
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        meta = reader.metadata or {}
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                pages.append(f"\n\n===== PAGE {page_no} =====\n\n{text}")
        relative = pdf_path.relative_to(ROOT)
        safe_stem = "__".join(relative.with_suffix("").parts)
        text_path = OUT / f"{safe_stem}.txt"
        text_path.write_text("".join(pages), encoding="utf-8")
        index.append(
            {
                "file": relative.as_posix(),
                "pages": len(reader.pages),
                "title": str(meta.get("/Title", "")),
                "author": str(meta.get("/Author", "")),
                "subject": str(meta.get("/Subject", "")),
                "text_file": str(text_path.relative_to(ROOT)),
            }
        )
    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
