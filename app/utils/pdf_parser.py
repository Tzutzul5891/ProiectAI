"""PDF parsing utilities.

The app can generate PDF statements (`app/utils/pdf_generator.py`). This module
extracts text from a PDF response so we can evaluate answers without any online
services (no OCR; only selectable/embedded text).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO


@dataclass(frozen=True)
class ParsedPDF:
    """Result of parsing a PDF document."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF byte payload.

    Notes:
      - Works only for PDFs that contain embedded/selectable text.
      - Scanned PDFs / handwriting require OCR (not included by design).
    """

    if not file_bytes:
        return ""

    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for PDF parsing. Install `pypdf` (see requirements.txt)."
        ) from exc

    reader = PdfReader(BytesIO(file_bytes))
    if getattr(reader, "is_encrypted", False):
        try:
            # Best-effort: many "encrypted" PDFs accept empty password.
            reader.decrypt("")  # type: ignore[attr-defined]
        except Exception as exc:
            raise ValueError("PDF-ul este criptat/parolat și nu poate fi citit.") from exc

    texts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            texts.append(page_text)

    return "\n\n".join(texts).replace("\x00", "")


def parse_pdf_bytes(pdf_bytes: bytes) -> ParsedPDF:
    """Extract text + a small metadata dict from a PDF payload."""

    text = extract_text_from_pdf(pdf_bytes)

    metadata: dict[str, str] = {}
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        meta = getattr(reader, "metadata", None)
        if isinstance(meta, dict):
            for k, v in meta.items():
                if k is None or v is None:
                    continue
                metadata[str(k)] = str(v)
        metadata["pages"] = str(len(getattr(reader, "pages", []) or []))
    except Exception:
        # Metadata is optional; don't fail parsing because of it.
        pass

    return ParsedPDF(text=text, metadata=metadata)
