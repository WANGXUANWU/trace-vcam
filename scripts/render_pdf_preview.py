"""Rasterize PDF pages so figure and page layout can be inspected visually."""

from __future__ import annotations

import argparse
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--pages", type=str, default="")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    wanted = (
        {int(item) for item in args.pages.split(",") if item.strip()}
        if args.pages
        else None
    )
    for path in args.inputs:
        document = pdfium.PdfDocument(str(path))
        for index in range(len(document)):
            if wanted is not None and index + 1 not in wanted:
                continue
            image = document[index].render(scale=args.scale).to_pil()
            suffix = "" if len(document) == 1 else f"_p{index + 1:02d}"
            target = args.output / f"{path.stem}{suffix}.png"
            image.save(target)
            print(target)


if __name__ == "__main__":
    main()
