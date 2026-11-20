#!/usr/bin/env python3
"""
Turn a markdown report into a PDF, with its figures embedded.

Images are written in the report as ordinary markdown images with paths
relative to the report file, for example
    ![Class distribution](../../figures/phase1/fig01_class_distribution.png)
so the same file reads correctly on GitHub and in the PDF.

Usage
    python src/reports/build_pdf.py reports/phase1/phase1_report.md reports/phase1.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 17pt; } h2 { font-size: 13.5pt; margin-top: 14pt; } h3 { font-size: 11.5pt; }
table { border-collapse: collapse; font-size: 8.5pt; }
th, td { border: 1px solid #999; padding: 2pt 4pt; }
th { background-color: #eeeeee; }
code { font-size: 8.5pt; }
pre { font-size: 8pt; background-color: #f4f4f4; padding: 4pt; }
img { width: 100%; }
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("markdown", type=Path)
    ap.add_argument("pdf", type=Path)
    args = ap.parse_args()

    text = args.markdown.read_text(encoding="utf-8")
    base = args.markdown.resolve().parent
    pattern = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")
    images = {m.group(2): (base / m.group(2)).resolve() for m in pattern.finditer(text)}
    missing = [ref for ref, path in images.items() if not path.exists()]
    if missing:
        print("images referenced in the report but not found:", *missing, sep="\n  ")
        return 1
    # The PDF engine resolves images inside one root folder and cannot follow
    # "../", so every image path is rewritten relative to a common root.
    root = Path(os.path.commonpath([base, *images.values()])) if images else base
    text = pattern.sub(lambda m: m.group(1) + images[m.group(2)].relative_to(root).as_posix() + m.group(3), text)
    base = root

    pdf = MarkdownPdf(toc_level=2, optimize=True)
    pdf.meta["title"] = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), args.markdown.stem)
    pdf.add_section(Section(text, root=str(base), paper_size="A4"), user_css=CSS)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(args.pdf))
    print(f"written: {args.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
