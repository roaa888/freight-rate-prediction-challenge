"""Render the editable report draft and scorer chart as a submission PDF."""

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAGE_BREAK = "<!-- PAGE BREAK -->"


def _plain(text: str) -> str:
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("`", "").replace("**", "")


def _render_text_page(lines: list[str], pdf: PdfPages, page_number: int) -> None:
    figure = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    x = 0.09
    y = 0.94
    for raw in lines:
        line = raw.strip()
        if not line:
            y -= 0.012
            continue
        if line.startswith("!["):
            continue
        if line.startswith("# "):
            text, size, weight, width, spacing = line[2:], 17, "bold", 58, 0.041
        elif line.startswith("## "):
            text, size, weight, width, spacing = line[3:], 12, "bold", 83, 0.031
        elif line.startswith("- "):
            text, size, weight, width, spacing = "- " + line[2:], 9.5, "normal", 100, 0.024
        else:
            text, size, weight, width, spacing = line, 9.5, "normal", 100, 0.024
        for part in textwrap.wrap(_plain(text), width=width, break_long_words=False):
            if y < 0.085:
                raise ValueError(f"Report page {page_number} overflows; add a page break")
            figure.text(x, y, part, fontsize=size, fontweight=weight,
                        color="#17333A", ha="left", va="top")
            y -= spacing
        if line.startswith("# ") or line.startswith("## "):
            y -= 0.009
    figure.text(0.09, 0.045, f"Freight Rate Prediction Challenge | Draft report | {page_number}",
                fontsize=8, color="#687A7F")
    pdf.savefig(figure)
    plt.close(figure)


def _render_chart_page(chart: Path, pdf: PdfPages, page_number: int) -> None:
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    figure.text(0.05, 0.93, "December 2025 predicted load rate", fontsize=18,
                fontweight="bold", color="#17333A")
    axis = figure.add_axes((0.05, 0.28, 0.9, 0.56))
    axis.imshow(plt.imread(chart))
    axis.axis("off")
    figure.text(0.05, 0.16,
                "Scorer output: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb",
                fontsize=10, color="#455A60")
    figure.text(0.05, 0.08, f"Freight Rate Prediction Challenge | Draft report | {page_number}",
                fontsize=8, color="#687A7F")
    pdf.savefig(figure)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=Path("report/report_draft.md"))
    parser.add_argument("--chart", type=Path,
                        default=Path("scorer_results/candidate_december.png"))
    parser.add_argument("--output", type=Path, default=Path("report/report_draft.pdf"))
    args = parser.parse_args()
    if not args.chart.is_file():
        raise FileNotFoundError(args.chart)
    pages = args.draft.read_text(encoding="utf-8").split(PAGE_BREAK)
    with PdfPages(args.output) as pdf:
        for number, page in enumerate(pages, start=1):
            _render_text_page(page.splitlines(), pdf, number)
        _render_chart_page(args.chart, pdf, len(pages) + 1)
    print(f"Created {args.output} with {len(pages) + 1} pages")


if __name__ == "__main__":
    main()
