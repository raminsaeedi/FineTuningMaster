"""Build retrieval chunks from the markdown guideline documents.

Each guideline file is split into chunks at headings (lines starting with '#'),
so one chunk is a heading plus its body text. Chunks get a stable id derived
from their source file and content, so rebuilding the KB does not reshuffle ids.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List


def _chunk_markdown(text: str, source: str) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    heading = ""
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if heading or body:
            content = (heading + "\n" + body).strip()
            digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
            chunks.append({
                "id": f"{source}_{digest}",
                "source": source,
                "heading": heading.lstrip("# ").strip(),
                "text": content,
            })

    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            heading = line.strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    # Drop chunks that are only a heading with no body.
    return [c for c in chunks if len(c["text"].split()) > 5]


def build_chunks(guidelines_dir: str | Path) -> List[Dict[str, str]]:
    """Read every .md file in ``guidelines_dir`` and return retrieval chunks."""
    guidelines_dir = Path(guidelines_dir)
    chunks: List[Dict[str, str]] = []
    for md_path in sorted(guidelines_dir.glob("*.md")):
        source = md_path.stem
        chunks.extend(_chunk_markdown(md_path.read_text(encoding="utf-8"), source))
    return chunks
