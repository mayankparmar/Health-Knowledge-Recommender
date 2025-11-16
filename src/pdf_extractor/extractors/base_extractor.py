"""
Base interface for content extractors.
"""

from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

from ..models import ExtractedContent, PDFDocument


class BaseExtractor(ABC):
    """Abstract base class for content extractors."""

    @abstractmethod
    def extract(self, document: PDFDocument) -> List[ExtractedContent]:
        """
        Extract content from a document.

        Args:
            document: PDFDocument metadata

        Returns:
            List of ExtractedContent objects
        """
        pass
