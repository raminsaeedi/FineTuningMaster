"""Regenerate the per-chapter reference lists from the central bibliography.

Each chapter under ``docs/thesis/chapters`` cites sources with LaTeX
``\\cite{key}`` commands. This script reads ``docs/thesis/references.bib``,
collects the keys each chapter actually cites, and rewrites the chapter's
"References Used in Chapter X" section so that the list always matches the
citations in the text and always uses one format.

The generated entry keeps the BibTeX key visible in backticks so that a reader
can move between the Markdown chapter and the ``.bib`` file, and so that the
Overleaf import stays checkable.

Usage::

    python experiments/scripts/build_chapter_reference_lists.py
    python experiments/scripts/build_chapter_reference_lists.py --check
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHAPTER_GLOB = "docs/thesis/chapters/*.md"
BIB_PATH = "docs/thesis/references.bib"
HEADING_RE = re.compile(r"(?m)^## References Used[^\n]*\n")
CITE_RE = re.compile(r"\\cite\{([^}]*)\}")


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    """Parse a flat BibTeX file into ``{key: {field: value}}``."""
    entries: dict[str, dict[str, str]] = {}
    for block in re.split(r"(?m)^(?=@)", text):
        header = re.match(r"@([a-zA-Z]+)\{([^,]+),", block)
        if not header:
            continue
        entry = {"_type": header.group(1).lower()}
        for field, braced, quoted, bare in re.findall(
            r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|([a-zA-Z]+))\s*,?",
            block[header.end():],
        ):
            entry[field.lower()] = (braced or quoted or bare).strip()
        entries[header.group(2).strip()] = entry
    return entries


def _clean(value: str) -> str:
    """Strip the BibTeX braces and escapes that only matter to LaTeX."""
    value = value.replace("\\&", "&")
    value = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", value)
    value = re.sub(r'\\"\{?([A-Za-z])\}?', r"\1", value)
    value = re.sub(r"\\'\{?([A-Za-z])\}?", r"\1", value)
    value = re.sub(r"\\`\{?([A-Za-z])\}?", r"\1", value)
    return value.replace("{", "").replace("}", "")


def format_authors(raw: str) -> str:
    """Render a BibTeX author list as 'Surname, I.' entries, truncated at six.

    A name wrapped in its own braces (``{Qwen Team}``) is a corporate author and
    is kept verbatim rather than split into surname and initials.
    """
    authors = [a.strip() for a in re.split(r"\s+and\s+(?![^{]*\})", raw) if a.strip()]
    rendered: list[str] = []
    for author in authors:
        if author.startswith("{") and author.endswith("}"):
            rendered.append(_clean(author))
            continue
        author = _clean(author)
        if "," in author:
            surname, given = [p.strip() for p in author.split(",", 1)]
        else:
            parts = author.split()
            surname, given = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (author, "")
        initials = " ".join(f"{p[0]}." for p in given.split() if p and p[0].isalpha())
        rendered.append(f"{surname}, {initials}".strip().rstrip(",") if initials else surname)
    if len(rendered) > 6:
        return ", ".join(rendered[:6]) + ", et al."
    if len(rendered) > 1:
        return ", ".join(rendered[:-1]) + ", & " + rendered[-1]
    return rendered[0] if rendered else "Unknown author"


def format_entry(key: str, e: dict[str, str]) -> str:
    """Render one bibliography entry as a single Markdown line."""
    author = format_authors(e.get("author", ""))
    year = _clean(e.get("year", "n.d."))
    title = _clean(e.get("title", "Untitled"))
    parts = [f"{author} ({year}). *{title}*."]

    venue = e.get("journal") or e.get("booktitle") or e.get("publisher") or e.get("howpublished")
    if venue:
        detail = _clean(venue)
        if e.get("volume"):
            detail += f", {_clean(e['volume'])}"
            if e.get("number"):
                detail += f"({_clean(e['number'])})"
        elif e.get("number"):
            detail += f", {_clean(e['number'])}"
        if e.get("pages"):
            detail += f", {_clean(e['pages']).replace('--', '–')}"
        parts.append(detail + ".")

    if e.get("edition"):
        parts.append(f"{_clean(e['edition'])}th ed.")
    if e.get("eprint") and not e.get("doi"):
        parts.append(f"arXiv:{_clean(e['eprint'])}.")
    if e.get("note"):
        parts.append(_clean(e["note"]) + ".")

    link = None
    if e.get("doi"):
        link = f"https://doi.org/{_clean(e['doi'])}"
    elif e.get("url"):
        link = _clean(e["url"])
    elif e.get("eprint"):
        link = f"https://arxiv.org/abs/{_clean(e['eprint'])}"
    if link:
        parts.append(link)

    return " ".join(parts) + f" [`{key}`]"


def chapter_number(path: Path) -> str:
    match = re.search(r"[Cc]hapter[_-]?(\d+)", path.name)
    return match.group(1) if match else "?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    bib = parse_bib((_PROJECT_ROOT / BIB_PATH).read_text(encoding="utf-8"))
    changed: list[str] = []
    missing: set[str] = set()

    for path_str in sorted(glob.glob(str(_PROJECT_ROOT / CHAPTER_GLOB))):
        path = Path(path_str)
        text = path.read_text(encoding="utf-8")
        body = HEADING_RE.split(text)[0].rstrip()

        keys: set[str] = set()
        for m in CITE_RE.finditer(body):
            keys |= {k.strip() for k in m.group(1).split(",") if k.strip()}
        unknown = keys - set(bib)
        missing |= unknown
        if unknown:
            print(f"  ! {path.name}: cite keys not in the bibliography: {sorted(unknown)}")

        number = chapter_number(path)
        lines = [
            "",
            "",
            f"## References Used in Chapter {number}",
            "",
            "Generated from `docs/thesis/references.bib` by "
            "`experiments/scripts/build_chapter_reference_lists.py`. "
            "The BibTeX key of each entry is given in brackets.",
            "",
        ]
        if keys:
            for key in sorted(keys, key=lambda k: (format_authors(bib[k].get("author", "")).lower(), k)):
                lines.append(format_entry(key, bib[key]))
                lines.append("")
        else:
            lines.append("This chapter cites no external sources.")
            lines.append("")

        new_text = body + "\n".join(lines)
        if new_text != text:
            changed.append(path.name)
            if not args.check:
                path.write_text(new_text, encoding="utf-8")
        print(f"{path.name:62} {len(keys):3} references")

    print()
    print(("would update: " if args.check else "updated: ") + (", ".join(changed) or "nothing"))
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
