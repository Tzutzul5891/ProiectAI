"""PDF parsing utilities (placeholder).

Current app flow generates PDFs (`app/utils/pdf_generator.py`) but does not yet
parse PDFs back into structured data. This module is the intended home for that
functionality once needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedPDF:
    """Result of parsing a PDF document."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def parse_pdf_bytes(pdf_bytes: bytes) -> ParsedPDF:
    """Extract text and basic metadata from a PDF.

    Not implemented yet. When needed, we can add an optional dependency like
    `pypdf` and implement a deterministic extractor here.
    """

    _ = pdf_bytes
    raise NotImplementedError(
        "PDF parsing is not implemented yet. Add a parser (e.g., `pypdf`) and implement `parse_pdf_bytes()`."
    )
