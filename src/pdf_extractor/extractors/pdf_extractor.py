"""
PDF content extractor with support for multiple libraries.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import uuid

from ..models import ExtractedContent, PDFDocument, ContentType, ExtractionMethod
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


class PDFExtractor(BaseExtractor):
    """
    Extracts structured content from PDF files.

    Supports multiple PDF libraries (pdfplumber, PyPDF2) and
    various extraction strategies.
    """

    def __init__(
        self,
        library: str = "auto",
        min_section_length: int = 50,
        min_paragraph_length: int = 20,
        extract_tips: bool = True
    ):
        """
        Initialize PDF extractor.

        Args:
            library: PDF library to use ("pdfplumber", "pypdf2", or "auto")
            min_section_length: Minimum characters for a section
            min_paragraph_length: Minimum characters for a paragraph
            extract_tips: Whether to extract tips/bullet points separately
        """
        self.min_section_length = min_section_length
        self.min_paragraph_length = min_paragraph_length
        self.extract_tips = extract_tips

        # Select PDF library
        if library == "auto":
            if HAS_PDFPLUMBER:
                self.library = "pdfplumber"
            elif HAS_PYPDF2:
                self.library = "pypdf2"
            else:
                raise ImportError(
                    "No PDF library available. Install pdfplumber or PyPDF2:\n"
                    "  pip install pdfplumber\n"
                    "  or\n"
                    "  pip install PyPDF2"
                )
        else:
            self.library = library

        logger.info(f"PDFExtractor initialized with library: {self.library}")

    def extract(self, document: PDFDocument) -> List[ExtractedContent]:
        """
        Extract content from a PDF document.

        Args:
            document: PDFDocument metadata

        Returns:
            List of ExtractedContent objects
        """
        file_path = Path(document.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Extracting content from {file_path.name} using {self.library}")

        # Extract raw sections
        if self.library == "pdfplumber":
            sections = self._extract_with_pdfplumber(file_path)
        else:
            sections = self._extract_with_pypdf2(file_path)

        logger.info(f"Extracted {len(sections)} raw sections from PDF")

        # Process sections into structured content
        extracted_content = self._process_sections(sections, document)

        logger.info(f"Created {len(extracted_content)} ExtractedContent objects")

        return extracted_content

    def _extract_with_pdfplumber(self, file_path: Path) -> List[Dict]:
        """Extract text using pdfplumber."""
        sections = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                # Split into paragraphs
                paragraphs = text.split('\n\n')

                for para in paragraphs:
                    para = para.strip()
                    if len(para) < self.min_paragraph_length:
                        continue

                    # Detect headings
                    is_heading = self._is_heading(para)

                    sections.append({
                        'page': page_num,
                        'text': para,
                        'type': 'heading' if is_heading else 'paragraph'
                    })

        return sections

    def _extract_with_pypdf2(self, file_path: Path) -> List[Dict]:
        """Extract text using PyPDF2."""
        sections = []

        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                # Split into paragraphs
                paragraphs = text.split('\n\n')

                for para in paragraphs:
                    para = para.strip()
                    if len(para) < self.min_paragraph_length:
                        continue

                    is_heading = self._is_heading(para)

                    sections.append({
                        'page': page_num,
                        'text': para,
                        'type': 'heading' if is_heading else 'paragraph'
                    })

        return sections

    def _is_heading(self, text: str) -> bool:
        """
        Detect if text is likely a heading.

        Heuristics:
        - All uppercase
        - Short length (< 100 chars)
        - Few words and no period at end
        - Ends with colon
        """
        if len(text) > 100:
            return False

        # All caps check
        if text.isupper():
            return True

        # Short without period
        if len(text.split()) <= 10 and not text.endswith('.'):
            return True

        # Ends with colon
        if text.endswith(':'):
            return True

        return False

    def _process_sections(
        self,
        raw_sections: List[Dict],
        document: PDFDocument
    ) -> List[ExtractedContent]:
        """
        Process raw sections into structured ExtractedContent objects.

        Args:
            raw_sections: List of raw section dictionaries
            document: PDFDocument metadata

        Returns:
            List of ExtractedContent objects
        """
        extracted_content = []
        current_heading = None
        current_hierarchy = []

        for section in raw_sections:
            if section['type'] == 'heading':
                # Update hierarchy
                current_heading = section['text']
                current_hierarchy = [current_heading]

                # Create section content
                content = ExtractedContent(
                    content_id=str(uuid.uuid4()),
                    doc_id=document.doc_id,
                    content_type=ContentType.SECTION,
                    title=current_heading,
                    text=current_heading,
                    page_number=section['page'],
                    section_hierarchy=current_hierarchy.copy(),
                    extraction_method=ExtractionMethod.RULE_BASED
                )
                extracted_content.append(content)

            else:  # paragraph
                para_text = section['text']

                # Check if this is a tip/bullet point
                if self.extract_tips and self._contains_tips(para_text):
                    # Extract individual tips
                    tips = self._extract_tips_from_text(para_text)
                    for tip_text in tips:
                        content = ExtractedContent(
                            content_id=str(uuid.uuid4()),
                            doc_id=document.doc_id,
                            content_type=ContentType.TIP,
                            title=self._create_title_from_text(tip_text),
                            text=tip_text,
                            page_number=section['page'],
                            section_hierarchy=current_hierarchy.copy(),
                            extraction_method=ExtractionMethod.RULE_BASED
                        )
                        extracted_content.append(content)
                else:
                    # Regular paragraph
                    if len(para_text) >= self.min_section_length:
                        content = ExtractedContent(
                            content_id=str(uuid.uuid4()),
                            doc_id=document.doc_id,
                            content_type=ContentType.PARAGRAPH,
                            title=current_heading or "Untitled Section",
                            text=para_text,
                            page_number=section['page'],
                            section_hierarchy=current_hierarchy.copy(),
                            extraction_method=ExtractionMethod.RULE_BASED
                        )
                        extracted_content.append(content)

        return extracted_content

    def _contains_tips(self, text: str) -> bool:
        """Check if text contains tips/bullet points."""
        # Look for common list markers
        markers = [
            r'^\s*[•\-\*]',  # Bullet points
            r'^\s*\d+\.',     # Numbered lists
            r'(?i)^\s*tip:',  # Tip: prefix
            r'(?i)^\s*try:',  # Try: prefix
        ]

        for marker in markers:
            if re.search(marker, text, re.MULTILINE):
                return True

        return False

    def _extract_tips_from_text(self, text: str) -> List[str]:
        """Extract individual tips from text containing bullet points."""
        tips = []
        lines = text.split('\n')
        current_tip = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts a new tip
            is_list_item = bool(re.match(
                r'^\s*([•\-\*]|\d+\.|\w+[:\)])\s+',
                line
            ))

            if is_list_item:
                # Save previous tip
                if current_tip:
                    tip_text = ' '.join(current_tip).strip()
                    # Clean up list markers
                    tip_text = re.sub(r'^[•\-\*\d\.]+\s*', '', tip_text)
                    tip_text = re.sub(r'^(tip|try):\s*', '', tip_text, flags=re.IGNORECASE)
                    if len(tip_text) >= self.min_paragraph_length:
                        tips.append(tip_text)

                # Start new tip
                current_tip = [line]
            elif current_tip:
                # Continue current tip
                current_tip.append(line)

        # Don't forget last tip
        if current_tip:
            tip_text = ' '.join(current_tip).strip()
            tip_text = re.sub(r'^[•\-\*\d\.]+\s*', '', tip_text)
            tip_text = re.sub(r'^(tip|try):\s*', '', tip_text, flags=re.IGNORECASE)
            if len(tip_text) >= self.min_paragraph_length:
                tips.append(tip_text)

        return tips

    def _create_title_from_text(self, text: str, max_words: int = 8) -> str:
        """Create a title from text by taking first few words."""
        words = text.split()[:max_words]
        title = ' '.join(words)
        if len(text.split()) > max_words:
            title += '...'
        return title
