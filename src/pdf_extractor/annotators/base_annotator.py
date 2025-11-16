"""
Base interface for content annotators.
"""

from abc import ABC, abstractmethod
from typing import List

from ..models import ExtractedContent, Annotation, PDFDocument


class BaseAnnotator(ABC):
    """Abstract base class for content annotators."""

    @abstractmethod
    def annotate(
        self,
        content: ExtractedContent,
        document: PDFDocument
    ) -> Annotation:
        """
        Annotate extracted content with FAST stages, capabilities, and topics.

        Args:
            content: ExtractedContent to annotate
            document: Source PDFDocument metadata

        Returns:
            Annotation object
        """
        pass

    @abstractmethod
    def annotate_batch(
        self,
        contents: List[ExtractedContent],
        document: PDFDocument
    ) -> List[Annotation]:
        """
        Annotate multiple content items in batch.

        Args:
            contents: List of ExtractedContent objects
            document: Source PDFDocument metadata

        Returns:
            List of Annotation objects
        """
        pass
