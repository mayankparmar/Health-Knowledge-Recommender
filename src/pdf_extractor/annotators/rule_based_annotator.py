"""
Rule-based content annotator using keyword matching and heuristics.
"""

import logging
import re
import uuid
from typing import List, Set, Dict, Tuple

from ..models import (
    ExtractedContent, Annotation, PDFDocument,
    ExtractionMethod, ImpairmentLevel
)
from ..loaders import FASTLoader, CapabilityLoader, MappingLoader
from .base_annotator import BaseAnnotator

logger = logging.getLogger(__name__)


class RuleBasedAnnotator(BaseAnnotator):
    """
    Annotates content using keyword matching and rule-based heuristics.

    Uses predefined dictionaries of keywords for:
    - FAST stages
    - Capabilities (ADLs/IADLs)
    - Topics
    - Target audiences
    """

    def __init__(
        self,
        fast_loader: FASTLoader,
        capability_loader: CapabilityLoader,
        mapping_loader: MappingLoader,
        min_confidence: float = 0.3
    ):
        """
        Initialize rule-based annotator.

        Args:
            fast_loader: Loaded FAST stage data
            capability_loader: Loaded capability data
            mapping_loader: Loaded mapping data
            min_confidence: Minimum confidence threshold (0.0-1.0)
        """
        self.fast_loader = fast_loader
        self.capability_loader = capability_loader
        self.mapping_loader = mapping_loader
        self.min_confidence = min_confidence

        # Build keyword dictionaries
        self._build_keyword_dictionaries()

    def _build_keyword_dictionaries(self):
        """Build keyword dictionaries for matching."""

        # FAST stage keywords
        self.fast_keywords = {
            'FAST-1': ['normal adult', 'no decline', 'intact cognition'],
            'FAST-2': ['normal older', 'subjective', 'age-related', 'forgetful'],
            'FAST-3': ['early alzheimer', 'early stage', 'mild cognitive impairment', 'mci',
                       'beginning', 'early dementia', 'initial symptoms'],
            'FAST-4': ['mild alzheimer', 'mild dementia', 'cannot live alone',
                       'needs supervision', 'moderate cognitive'],
            'FAST-5': ['moderate alzheimer', 'moderate dementia', 'middle stage',
                       'cannot select clothing', 'needs assistance daily'],
            'FAST-6a': ['dressing difficulty', 'dressing impairment', 'cannot dress'],
            'FAST-6b': ['bathing difficulty', 'cannot bathe', 'shower assistance'],
            'FAST-6c': ['toileting mechanics', 'toilet assistance', 'hygiene difficulty'],
            'FAST-6d': ['urinary incontinence', 'bladder control', 'urine leakage'],
            'FAST-6e': ['fecal incontinence', 'bowel incontinence', 'bowel control'],
            'FAST-7a': ['limited speech', 'few words', 'speech decline', 'verbal decline'],
            'FAST-7b': ['single word', 'one word', 'minimal speech'],
            'FAST-7c': ['cannot walk', 'non-ambulatory', 'wheelchair bound', 'lost mobility'],
            'FAST-7d': ['cannot sit', 'sitting difficulty', 'cannot sit unsupported'],
            'FAST-7e': ['cannot smile', 'lost smile', 'facial expression'],
            'FAST-7f': ['cannot hold head', 'head support', 'end stage', 'terminal', 'palliative']
        }

        # Broad stage groups
        self.stage_group_keywords = {
            'early': ['FAST-3', 'FAST-4'],
            'moderate': ['FAST-4', 'FAST-5', 'FAST-6a', 'FAST-6b'],
            'moderately severe': ['FAST-6a', 'FAST-6b', 'FAST-6c', 'FAST-6d', 'FAST-6e'],
            'severe': ['FAST-7a', 'FAST-7b', 'FAST-7c', 'FAST-7d', 'FAST-7e', 'FAST-7f'],
            'late': ['FAST-6d', 'FAST-6e', 'FAST-7a', 'FAST-7b', 'FAST-7c'],
            'end-stage': ['FAST-7d', 'FAST-7e', 'FAST-7f']
        }

        # Capability keywords (based on capability names and descriptions)
        self.capability_keywords = {
            'ADL-1': ['bath', 'bathing', 'shower', 'wash', 'washing', 'hygiene', 'clean'],
            'ADL-2': ['dress', 'dressing', 'clothing', 'clothes', 'outfit', 'wearing'],
            'ADL-3': ['toilet', 'toileting', 'bathroom', 'urinate', 'bowel'],
            'ADL-4': ['transfer', 'transferring', 'mobility', 'walking', 'moving', 'wheelchair', 'ambulating'],
            'ADL-5': ['continence', 'incontinence', 'bladder', 'bowel control', 'accidents'],
            'ADL-6': ['feed', 'feeding', 'eating', 'meal', 'nutrition', 'swallow', 'dining'],
            'IADL-1': ['phone', 'telephone', 'calling', 'call', 'dial'],
            'IADL-2': ['shopping', 'shop', 'groceries', 'store', 'purchase'],
            'IADL-3': ['cooking', 'food preparation', 'meal prep', 'kitchen', 'recipe', 'prepare meal'],
            'IADL-4': ['housekeeping', 'cleaning', 'chores', 'tidy', 'household', 'maintain home'],
            'IADL-5': ['laundry', 'washing clothes', 'wash clothes', 'clothing care'],
            'IADL-6': ['transportation', 'transport', 'driving', 'drive', 'travel', 'car', 'bus'],
            'IADL-7': ['medication', 'medicine', 'pills', 'drugs', 'prescription', 'dose'],
            'IADL-8': ['financial', 'finances', 'money', 'bills', 'banking', 'budget', 'pay', 'fraud']
        }

        # Topic keywords
        self.topic_keywords = {
            'Safety': ['safety', 'safe', 'danger', 'risk', 'hazard', 'prevent', 'protection'],
            'Communication': ['communication', 'communicate', 'talking', 'speaking', 'language', 'conversation'],
            'Behavioral Symptoms': ['behavior', 'behaviour', 'agitation', 'aggression', 'wandering',
                                   'anxiety', 'depression', 'mood'],
            'Medication Management': ['medication', 'medicine', 'pills', 'prescription'],
            'Legal Planning': ['legal', 'power of attorney', 'poa', 'advance directive', 'will', 'guardian'],
            'Financial Management': ['financial', 'money', 'bills', 'fraud', 'scam', 'banking'],
            'Caregiver Support': ['caregiver', 'carer', 'respite', 'stress', 'support', 'self-care', 'burnout'],
            'Activities': ['activities', 'hobbies', 'engagement', 'social', 'recreation', 'exercise'],
            'Nutrition': ['nutrition', 'eating', 'diet', 'food', 'hydration', 'meal', 'appetite'],
            'Personal Care': ['personal care', 'bathing', 'dressing', 'hygiene', 'grooming'],
            'Diagnosis': ['diagnosis', 'screening', 'assessment', 'evaluation', 'test'],
            'Treatment': ['treatment', 'therapy', 'intervention', 'clinical'],
            'End-of-Life Care': ['end of life', 'palliative', 'hospice', 'comfort care', 'terminal'],
            'Daily Routine': ['routine', 'schedule', 'daily', 'day-to-day', 'activities of daily living']
        }

        # Audience keywords
        self.audience_keywords = {
            'Patient': ['patient', 'person with dementia', 'individual', 'you are', 'your symptoms'],
            'Caregiver': ['caregiver', 'carer', 'family', 'caring for', 'look after',
                         'help them', 'support them', 'their needs'],
            'Both': ['everyone', 'all', 'family and patient', 'together']
        }

    def annotate(self, content: ExtractedContent, document: PDFDocument) -> Annotation:
        """Annotate a single content item."""
        text = (content.title + " " + content.text).lower()

        # Detect FAST stages
        fast_stages, fast_confidence = self._detect_fast_stages(text)

        # Detect capabilities
        capabilities, capability_confidence = self._detect_capabilities(text, fast_stages)

        # Detect topics
        topics = self._detect_topics(text)

        # Detect keywords
        keywords = self._extract_keywords(text)

        # Detect target audience
        target_audience = self._detect_audience(text, document)

        annotation = Annotation(
            annotation_id=str(uuid.uuid4()),
            content_id=content.content_id,
            fast_stages=fast_stages,
            fast_confidence=fast_confidence,
            capabilities=capabilities,
            capability_confidence=capability_confidence,
            topics=topics,
            keywords=keywords,
            target_audience=target_audience,
            annotation_method=ExtractionMethod.RULE_BASED
        )

        return annotation

    def annotate_batch(
        self,
        contents: List[ExtractedContent],
        document: PDFDocument
    ) -> List[Annotation]:
        """Annotate multiple content items."""
        return [self.annotate(content, document) for content in contents]

    def _detect_fast_stages(self, text: str) -> Tuple[List[str], float]:
        """
        Detect FAST stages mentioned in text.

        Returns:
            Tuple of (list of stage codes, confidence score)
        """
        detected_stages = set()
        keyword_matches = 0
        total_checks = 0

        # Check specific stage keywords
        for stage_code, keywords in self.fast_keywords.items():
            total_checks += 1
            for keyword in keywords:
                if keyword in text:
                    detected_stages.add(stage_code)
                    keyword_matches += 1
                    break

        # Check broad stage groups
        for group_name, stages in self.stage_group_keywords.items():
            if group_name in text:
                detected_stages.update(stages)
                keyword_matches += 1

        # Calculate confidence based on keyword matches
        if not detected_stages:
            # Default to broad range if no specific detection
            detected_stages = {'FAST-3', 'FAST-4', 'FAST-5'}  # Common middle stages
            confidence = self.min_confidence
        else:
            confidence = min(0.95, 0.5 + (keyword_matches / max(total_checks, 1)) * 0.5)

        # Sort stages by stage number
        sorted_stages = sorted(
            list(detected_stages),
            key=lambda s: self.fast_loader.get_stage(s).stage_number
        )

        return sorted_stages, confidence

    def _detect_capabilities(
        self,
        text: str,
        fast_stages: List[str]
    ) -> Tuple[List[str], float]:
        """
        Detect capabilities mentioned in text.

        Args:
            text: Text to analyze
            fast_stages: Detected FAST stages (for mapping context)

        Returns:
            Tuple of (list of capability IDs, confidence score)
        """
        detected_capabilities = set()
        keyword_matches = 0

        # Keyword-based detection
        for cap_id, keywords in self.capability_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected_capabilities.add(cap_id)
                    keyword_matches += 1
                    break

        # Enhance with FAST mapping
        # If we detected FAST stages, add likely impaired capabilities
        if fast_stages:
            for stage in fast_stages:
                impaired_caps = self.mapping_loader.get_impaired_capabilities_for_stage(
                    stage,
                    min_impairment=ImpairmentLevel.REQUIRES_PROMPTING
                )
                # Add with lower priority (only if some keyword match exists)
                if keyword_matches > 0:
                    detected_capabilities.update(impaired_caps[:3])  # Top 3 impaired

        # Calculate confidence
        if not detected_capabilities:
            confidence = 0.0
        else:
            # Higher confidence if keyword matches, lower if only from mapping
            base_confidence = min(0.9, keyword_matches * 0.2 + 0.3)
            confidence = base_confidence

        return sorted(list(detected_capabilities)), confidence

    def _detect_topics(self, text: str) -> List[str]:
        """Detect topics mentioned in text."""
        detected_topics = []

        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected_topics.append(topic)
                    break

        # Return top 5 topics
        return detected_topics[:5]

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text.

        Simple approach: extract capitalized words and domain-specific terms.
        """
        keywords = set()

        # Find capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        keywords.update(capitalized)

        # Extract medical/domain terms (simplified)
        domain_terms = [
            'dementia', 'alzheimer', 'memory', 'cognitive', 'caregiver',
            'medication', 'safety', 'diagnosis', 'treatment', 'symptom'
        ]
        for term in domain_terms:
            if term in text:
                keywords.add(term.capitalize())

        return sorted(list(keywords))[:max_keywords]

    def _detect_audience(self, text: str, document: PDFDocument) -> str:
        """
        Detect target audience.

        Args:
            text: Text to analyze
            document: Document metadata (may contain audience info)

        Returns:
            Target audience string
        """
        # First check document metadata
        if document.target_audience:
            return document.target_audience

        # Otherwise, use keyword matching
        audience_scores = {}
        for audience, keywords in self.audience_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            audience_scores[audience] = score

        # Return audience with highest score, default to "Both"
        max_audience = max(audience_scores.items(), key=lambda x: x[1])
        if max_audience[1] > 0:
            return max_audience[0]
        else:
            return "Both"
