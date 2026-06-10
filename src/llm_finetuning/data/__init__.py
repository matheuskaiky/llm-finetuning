"""Dataset loaders. Importing this module registers the built-in loaders."""

from __future__ import annotations

from .pdf_loader import PdfToTextLoader, normalize_text

__all__ = ["PdfToTextLoader", "normalize_text"]
