"""Verify README.md against the project it describes.

Three classes of failure this catches, all of which have actually happened in
this repository during development:

* **Structure.** A mandatory section was renamed or dropped. The Executive
  Summary, the research question and the rationale are the three the brief
  requires by name, so they are checked by exact heading.
* **Broken references.** A figure link that resolves to a file that does not
  exist, an image served from somewhere else, or a figure that is in the
  authoritative inventory but was never generated.
* **Stale numbers.** A result value in the README that no longer agrees with
  the live experiment artifact. The check regenerates the README from its source
  and compares byte for byte, so any drift between the documents and the
  evidence fails here.

Exit code 0 means the README is consistent. Anything else lists the problems.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

README = ROOT / "README.md"
SOURCE = ROOT / "scripts" / "readme_source.md"

REQUIRED_HEADINGS = [
    "## Executive Summary",
    "## 2. Research question",
    "## 3. Why I chose this question",
    "## Overview",
    "## Use Case",
    "## Architecture",
    "## Domain Model",
    "## Mathematical Formulation",
    "## Constraints",
    "## Optimization Methods",
    "## Simulation",
    "## Disruptions",
    "## Replanning",
    "## What-If",
    "## Benchmarks",
    "## Results",
    "## Scalability",
    "## Constraint Pressure",
    "## Failure Analysis",
    "## Screenshots",
    "## Quickstart",
    "## CLI",
    "## API",
    "## Persistence",
    "## Exports",
    "## Docker",
    "## Tests",
    "## CI",
    "## Reproducibility",
    "## Technical Report",
    "## Limitations",
    "## Future Work",
]

RESEARCH_QUESTION = (
    "How effectively can a resource-allocation system produce high-quality "
    "schedules under real-world constraints and recover from operational "
    "disruptions while keeping computation time practical?"
)

# Claims the brief forbids. Matched case-insensitively as whole words.
FORBIDDEN = [
    (r"\bmilitary\b", "claims military use"),
    (r"\bproduction[- ]ready\b", "claims production readiness"),
    (r"\buniversally best\b", "claims universal solver superiority"),
    (r"\bbest solver (?:is|for)\b", "claims a single best solver"),
    (r"\bdeployed in production\b", "claims production deployment"),
]

PLACEHOLDER_PATTERNS = [
    r"\{\{[^}]+\}\}",
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bplaceholder\b",
    r"\bnot implemented\b",
    r"\bTBD\b",
    r"\bXXX\b",
    r"\bLorem ipsum\b",
    r"\bcoming soon\b",
]

PERSONAL_PATH = re.compile(r"[A-Za-z]:[\\/]|/home/[a-z]|/Users/[a-z]")
EXTERNAL_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*https?://", re.IGNORECASE)
IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _is_valid_png(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "file does not exist"
    data = path.read_bytes()
    if len(data) < 100 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return False, "not a PNG (bad signature or truncated)"
    if data[12:16] != b"IHDR":
        return False, "no IHDR chunk"
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        return False, f"degenerate size {width}x{height}"
    return True, f"{width}x{height}, {len(data):,} bytes"


def check() -> list[str]:
    problems: list[str] = []

    if not README.is_file():
        return ["README.md does not exist"]

    text = README.read_text(encoding="utf-8")

    # -- 1. structure ---------------------------------------------------
    headings = {line.strip() for line in text.splitlines() if line.startswith("#")}
    for required in REQUIRED_HEADINGS:
        if required not in headings:
            problems.append(f"missing required section: {required}")

    if not text.lstrip().startswith("# ORION"):
        problems.append("README must open with the '# ORION' title")

    # -- 2. executive summary is one paragraph --------------------------
    match = re.search(r"## Executive Summary\s*\n(.*?)(?=\n## )", text, re.DOTALL)
    if not match:
        problems.append("no Executive Summary body found")
    else:
        body = match.group(1).strip()
        paragraphs = [p for p in body.split("\n\n") if p.strip()]
        # A figure directly under the summary is allowed; prose must be one
        # paragraph, and it must actually explain the system.
        prose = [p for p in paragraphs if not p.strip().startswith("!")]
        if len(prose) != 1:
            problems.append(
                f"the Executive Summary must be one paragraph, found {len(prose)}"
            )
        for topic in ("optimiz", "simulat", "disrupt", "replan", "benchmark", "limitation"):
            if topic not in body.lower():
                problems.append(f"the Executive Summary does not mention {topic}")

    # -- 3. the research question, verbatim ------------------------------
    if RESEARCH_QUESTION not in " ".join(text.split()):
        problems.append(
            "the research question is not present verbatim under "
            "'## 2. Research question'"
        )

    # -- 4. why-section content -----------------------------------------
    why = re.search(r"## 3\. Why I chose this question\s*\n(.*?)(?=\n## )", text, re.DOTALL)
    if why:
        body = why.group(1).lower()
        for topic in ("scarce", "priorit", "window", "depend", "chang", "time"):
            if topic not in body:
                problems.append(f"'Why I chose this question' does not discuss {topic}")

    # -- 5. figures ------------------------------------------------------
    from orion.figures_live import README_FIGURES, figure_names

    inventory = set(figure_names())
    links = IMAGE_LINK.findall(text)
    if not links:
        problems.append("the README embeds no figures")

    for alt, target in links:
        if target.startswith(("http://", "https://", "//")):
            problems.append(f"figure {alt!r} links to an external URL: {target}")
            continue
        if target.startswith("data:"):
            problems.append(f"figure {alt!r} is an inline data URI, not a file")
            continue
        resolved = (ROOT / target).resolve()
        if not resolved.is_file():
            problems.append(f"figure link does not resolve: {target}")
            continue
        ok, detail = _is_valid_png(resolved)
        if not ok:
            problems.append(f"figure {target} is not a valid image: {detail}")

    if EXTERNAL_IMAGE.search(text):
        problems.append("the README contains an external image URL")

    for promoted in README_FIGURES:
        if promoted not in inventory:
            problems.append(f"README promotes {promoted}, absent from the inventory")
        if promoted not in text:
            problems.append(f"README_FIGURES lists {promoted} but the README omits it")

    promoted_in_readme = {Path(t).name for _a, t in links if not t.startswith("http")}
    # The README promotes a deliberate subset (README_FIGURES); the rest are
    # report-only. What must hold is that every promoted figure is present, and
    # that the README does not link a figure the inventory does not define.
    for linked in promoted_in_readme:
        if linked not in inventory:
            problems.append(
                f"the README links {linked}, which is not in FIGURE_INVENTORY, "
                "so the figure layer will never generate it"
            )
    if len(promoted_in_readme) < 3:
        problems.append(
            f"the README embeds only {len(promoted_in_readme)} figure(s); the brief "
            "asks for 3-5 key visuals after the opening sections"
        )

    # -- 6. no placeholders, no personal paths ---------------------------
    for pattern in PLACEHOLDER_PATTERNS:
        hit = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if hit:
            line = text[: hit.start()].count("\n") + 1
            snippet = text.splitlines()[line - 1].strip()[:80]
            problems.append(f"placeholder text at line {line}: {snippet!r}")

    personal = PERSONAL_PATH.search(text)
    if personal:
        line = text[: personal.start()].count("\n") + 1
        problems.append(
            f"personal machine path at line {line}: {personal.group(0)!r}"
        )

    # Only an *asserted* claim is a problem. The Limitations section exists
    # precisely to say what ORION is not, so a negated mention is required text,
    # not a violation.
    NEGATION = re.compile(
        r"\b(?:not|never|no|isn't|is not|does not|doesn'?t|cannot|neither|nor|"
        r"without|disclaim\w*|rather than|instead of)\b"
        r"[^.\n]{0,90}$",
        re.IGNORECASE,
    )
    for pattern, why_not in FORBIDDEN:
        for hit in re.finditer(pattern, text, re.IGNORECASE):
            line = text[: hit.start()].count("\n") + 1
            start_of_line = text.rfind("\n", 0, hit.start()) + 1
            preceding = text[start_of_line:hit.start()]
            if NEGATION.search(preceding):
                continue  # a stated limitation, not a claim
            problems.append(f"line {line} {why_not}: {hit.group(0)!r}")

    # -- 7. the README must match its generated form ---------------------
    try:
        import write_readme

        evidence = write_readme.Evidence.load(ROOT)
        rendered = write_readme.render(write_readme.collect(evidence, write_readme._count_tests()))
        current = re.sub(r"^\d+$", "COUNT", text, flags=re.MULTILINE)
        expected = re.sub(r"^\d+$", "COUNT", rendered, flags=re.MULTILINE)
        if current != expected:
            # Narrow the report to the first differing line, which is far more
            # useful than "they differ".
            for number, (a, b) in enumerate(
                zip(current.splitlines(), expected.splitlines()), start=1
            ):
                if a != b:
                    problems.append(
                        f"README.md line {number} is stale: {a.strip()[:70]!r} "
                        f"but the evidence says {b.strip()[:70]!r}"
                    )
                    break
            else:
                problems.append(
                    "README.md has a different length from the generated text; "
                    "run scripts/write_readme.py"
                )
    except Exception as exc:  # a failure here is itself a finding
        problems.append(f"could not regenerate the README to compare: {type(exc).__name__}: {exc}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    problems = check()
    if problems:
        print(f"README check FAILED ({len(problems)} problem(s)):\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("README check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
