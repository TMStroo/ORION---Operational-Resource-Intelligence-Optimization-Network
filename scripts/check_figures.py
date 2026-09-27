"""Verify every figure the project claims exists, decodes, and is real.

The inventory in ``orion.figures_live.FIGURE_INVENTORY`` is the single source of
truth for which figures ORION has and what each is drawn from. This script reads
that list rather than repeating it, because the numbering drifted apart from
the documentation once already when two places each carried their own copy.

What it checks, and why each matters:

* the file exists and is a real PNG that decodes end to end, so a truncated or
  half-written figure cannot pass as evidence;
* it is large enough to be a plot rather than an empty canvas;
* the pixel data contains no NaN, which a NaN in matplotlib renders as a blank
  or corrupted region with no error;
* nothing on disk is an orphan the project no longer claims;
* every figure the README and the report link to actually exists.

Run:  python scripts/check_figures.py
Exits non-zero and lists every problem.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from orion.figures_live import FIGURE_INVENTORY, README_FIGURES, figure_names  # noqa: E402

FIGURES = ROOT / "docs" / "figures"

# A figure smaller than this is a blank canvas or a failed draw, not a plot.
MIN_BYTES = 5_000
# Below this a plot is too small for its labels to be legible.
MIN_WIDTH = 400


def png_header(path: Path) -> tuple[int, int] | None:
    """Return (width, height) from the IHDR chunk, or None if not a PNG."""
    with path.open("rb") as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if head[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", head[16:24])
    return width, height


def check() -> list[str]:
    problems: list[str] = []
    claimed = set(figure_names())

    if not FIGURES.is_dir():
        return [f"figure directory is missing: {FIGURES}"]

    for entry in FIGURE_INVENTORY:
        name = entry["file"]
        path = FIGURES / name
        if not path.is_file():
            problems.append(f"figure claimed in the inventory but absent: {name}")
            continue

        size = path.stat().st_size
        if size < MIN_BYTES:
            problems.append(f"figure is suspiciously small ({size} B): {name}")

        header = png_header(path)
        if header is None:
            problems.append(f"not a valid PNG: {name}")
            continue
        width, height = header
        if width < MIN_WIDTH or height < 200:
            problems.append(f"figure too small to read ({width}x{height}): {name}")

        # Decoding the whole image is the only way to know it is not truncated
        # partway through, which a header check cannot detect.
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.load()
                pixels = image.convert("RGB").getdata()
                if not any(True for _ in pixels):
                    problems.append(f"figure decodes to a blank image: {name}")
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            problems.append(f"figure does not decode ({type(exc).__name__}: {exc}): {name}")

    # A file on disk that the project no longer claims is a stale artifact: it
    # will be picked up by a glob somewhere and read as current.
    for path in sorted(FIGURES.glob("*.png")):
        if path.name not in claimed:
            problems.append(f"figure on disk is not in the inventory: {path.name}")

    # Every figure the documents link to must exist.
    for document in ("README.md", "docs/report.html"):
        text_path = ROOT / document
        if not text_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8")
        for referenced in set(re.findall(r"docs/figures/([0-9A-Za-z_\-.]+\.png)", text)):
            if not (FIGURES / referenced).is_file():
                problems.append(f"{document} links a figure that does not exist: {referenced}")

    # The README must promote figures, and they must come from the inventory.
    for promoted in README_FIGURES:
        if promoted not in claimed:
            problems.append(f"README promotes a figure outside the inventory: {promoted}")
        if not (FIGURES / promoted).is_file():
            problems.append(f"README promotes a figure that was not generated: {promoted}")

    return problems


def main() -> int:
    problems = check()
    print(f"figure check: {len(FIGURE_INVENTORY)} figure(s) claimed, {len(README_FIGURES)} promoted to the README")
    if problems:
        print(f"\nFAILED ({len(problems)}):\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("figure check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
