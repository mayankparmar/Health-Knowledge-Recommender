"""
Core data models for the PDF extraction system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class CapabilityType(Enum):
    """Type of functional capability."""
    ADL = "ADL"
    IADL = "IADL"


class ImpairmentLevel(Enum):
    """Level of functional impairment."""
    INDEPENDENT = "Independent"
    REQUIRES_PROMPTING = "Requires Prompting"
    REQUIRES_ASSISTANCE = "Requires Assistance"
    IMPAIRED = "Impaired"
    DEPENDENT = "Dependent"


class ExtractionMethod(Enum):
    """Method used for extraction."""
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"
    HYBRID = "hybrid"


class ContentType(Enum):
    """Type of extracted content."""
    SECTION = "section"
    TIP = "tip"
    GUIDELINE = "guideline"
    PARAGRAPH = "paragraph"


@dataclass
class FASTStage:
    """FAST (Functional Assessment Staging Tool) stage."""
    stage_code: str  # e.g., "FAST-3", "FAST-6a"
    stage_name: str
    clinical_characteristics: str
    cognition: str
    adl_status: str
    iadl_status: str
    typical_duration: Optional[str] = None
    care_needs: Optional[str] = None

    @property
    def stage_number(self) -> float:
        """Extract numeric stage for ordering."""
        stage_str = self.stage_code.replace("FAST-", "")
        # Handle substages like "6a", "7f"
        if stage_str[-1].isalpha():
            base = float(stage_str[:-1])
            substage = ord(stage_str[-1]) - ord('a')
            return base + (substage * 0.1)
        return float(stage_str)


@dataclass
class Capability:
    """ADL or IADL capability."""
    capability_id: str  # e.g., "ADL-1", "IADL-3"
    capability_type: CapabilityType
    name: str
    description: str
    independence_criteria: Optional[str] = None
    dependence_indicators: Optional[str] = None
    icf_code_primary: Optional[str] = None
    icf_code_secondary: Optional[str] = None
    icf_category: Optional[str] = None


@dataclass
class FASTCapabilityMapping:
    """Mapping between FAST stage and capability impairment."""
    fast_stage: str
    capability_id: str
    capability_name: str
    capability_type: CapabilityType
    impairment_level: ImpairmentLevel
    clinical_notes: Optional[str] = None
    information_needs: Optional[str] = None
    icf_code: Optional[str] = None


@dataclass
class PDFDocument:
    """Metadata about a PDF document."""
    doc_id: str
    name: str
    file_path: str
    url: Optional[str] = None
    source_organization: Optional[str] = None
    target_audience: Optional[str] = None  # "Patient", "Caregiver", "Both"
    publication_date: Optional[str] = None
    language: str = "en"


@dataclass
class ExtractedContent:
    """Content extracted from a PDF."""
    content_id: str
    doc_id: str
    content_type: ContentType
    title: str
    text: str
    page_number: Optional[int] = None
    section_hierarchy: Optional[List[str]] = None

    # Metadata
    extraction_method: ExtractionMethod = ExtractionMethod.RULE_BASED
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content_id": self.content_id,
            "doc_id": self.doc_id,
            "content_type": self.content_type.value,
            "title": self.title,
            "text": self.text,
            "page_number": self.page_number,
            "section_hierarchy": self.section_hierarchy,
            "extraction_method": self.extraction_method.value,
            "extraction_timestamp": self.extraction_timestamp
        }


@dataclass
class Annotation:
    """Annotation linking content to FAST stages and capabilities."""
    annotation_id: str
    content_id: str

    # FAST staging
    fast_stages: List[str]  # e.g., ["FAST-3", "FAST-4"]
    fast_confidence: float  # 0.0 to 1.0

    # Capabilities
    capabilities: List[str]  # capability IDs, e.g., ["ADL-1", "IADL-3"]
    capability_confidence: float

    # Topics and keywords
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Target audience
    target_audience: Optional[str] = None  # "Patient", "Caregiver", "Both"

    # Annotation metadata
    annotation_method: ExtractionMethod = ExtractionMethod.RULE_BASED
    annotation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    annotator_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "content_id": self.content_id,
            "fast_stages": self.fast_stages,
            "fast_confidence": self.fast_confidence,
            "capabilities": self.capabilities,
            "capability_confidence": self.capability_confidence,
            "topics": self.topics,
            "keywords": self.keywords,
            "target_audience": self.target_audience,
            "annotation_method": self.annotation_method.value,
            "annotation_timestamp": self.annotation_timestamp,
            "annotator_notes": self.annotator_notes
        }


@dataclass
class AnnotatedContent:
    """Extracted content with its annotations."""
    content: ExtractedContent
    annotation: Annotation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content.to_dict(),
            "annotation": self.annotation.to_dict()
        }
