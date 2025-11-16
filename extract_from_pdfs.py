#!/usr/bin/env python3
"""
PDF Information Extraction Script
Health Knowledge Recommender Project - WP-03

Extracts comprehensive information from Alzheimer's Society UK PDFs:
1. The dementia guide
2. Caring for a person with dementia: A practical guide

Extracts both section-level information and detailed tips.
Tags with FAST stages, capabilities, audience, and topics.
"""

import json
import csv
import os
import sys
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# PDF processing
try:
    import pdfplumber
    PDF_LIBRARY = 'pdfplumber'
except ImportError:
    try:
        import PyPDF2
        PDF_LIBRARY = 'PyPDF2'
    except ImportError:
        print("ERROR: No PDF library found. Install pdfplumber or PyPDF2:")
        print("  pip install pdfplumber")
        sys.exit(1)

# Load environment variables
load_dotenv()

# ============================================================================
# LLM Provider (reuse from wp03_extract_information.py)
# ============================================================================

class LLMProvider:
    """LLM provider for content analysis and extraction."""

    def __init__(self, config: Dict):
        self.config = config
        self.provider = config['llm']['provider']
        self.model = config['llm']['model']
        self.temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_tokens']

        # Get API key
        provider_config = config.get('llm_providers', {}).get(self.provider, {})
        if not provider_config:
            # Simple config without llm_providers section
            api_key_var = config['llm'].get('api_key_env_var')
        else:
            api_key_var = provider_config.get('api_key_env_var')

        self.api_key = os.getenv(api_key_var) if api_key_var else None
        self.client = None

        if self.provider != 'none' and self.api_key:
            self._init_client()

    def _init_client(self):
        """Initialize LLM client."""
        if self.provider == 'anthropic':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logging.info(f"Initialized Anthropic client with model {self.model}")
            except ImportError:
                logging.warning("anthropic package not installed. Run: pip install anthropic")

        elif self.provider == 'openai':
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logging.info(f"Initialized OpenAI client with model {self.model}")
            except ImportError:
                logging.warning("openai package not installed. Run: pip install openai")

        elif self.provider == 'gemini':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                logging.info(f"Initialized Gemini client with model {self.model}")
            except ImportError:
                logging.warning("google-generativeai package not installed. Run: pip install google-generativeai")

    def analyze_content(self, text: str, prompt_type: str = "extract") -> str:
        """Analyze content using LLM or return empty if not available."""
        if not self.client or self.provider == 'none':
            return ""

        try:
            if prompt_type == "extract":
                prompt = self._build_extraction_prompt(text)
            elif prompt_type == "classify_stage":
                prompt = self._build_stage_classification_prompt(text)
            else:
                prompt = text

            # Call appropriate LLM
            if self.provider == 'anthropic':
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text

            elif self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content

            elif self.provider == 'gemini':
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': self.temperature,
                        'max_output_tokens': self.max_tokens
                    }
                )
                return response.text

        except Exception as e:
            logging.error(f"LLM analysis failed: {e}")
            return ""

    def _build_extraction_prompt(self, text: str) -> str:
        """Build prompt for extracting detailed information."""
        return f"""Analyze this section from an Alzheimer's Society UK guide and extract key information.

For this content, identify:
1. Main topic/theme
2. FAST stages this applies to (FAST-1 through FAST-7f, or ranges like "FAST-3,FAST-4,FAST-5")
3. Related ADL/IADL capabilities (if any)
4. 2-5 specific, actionable tips or pieces of information
5. Whether this is for Patient, Caregiver, or Both

CONTENT:
{text[:3000]}

Provide your analysis in this format:
TOPIC: [main topic]
FAST_STAGES: [applicable stages]
CAPABILITY: [ADL/IADL or "General"]
AUDIENCE: [Patient/Caregiver/Both]
TIPS:
- [tip 1]
- [tip 2]
- [tip 3]
"""

    def _build_stage_classification_prompt(self, text: str) -> str:
        """Build prompt for FAST stage classification."""
        return f"""Classify which FAST dementia stages this content applies to.

FAST STAGES REFERENCE:
- FAST-1,FAST-2: Normal/subjective decline
- FAST-3: Early Alzheimer's (mild cognitive impairment, IADL beginning to be affected)
- FAST-4: Mild Alzheimer's (IADL impaired, ADL prompting begins, cannot survive alone)
- FAST-5: Moderate Alzheimer's (all IADL dependent, ADL prompting intensive)
- FAST-6a: Moderately severe (dressing impairment)
- FAST-6b: Moderately severe (bathing impairment)
- FAST-6c: Moderately severe (toileting mechanics fail)
- FAST-6d: Moderately severe (urinary incontinence)
- FAST-6e: Moderately severe (fecal incontinence)
- FAST-7a: Severe (speech 1-5 words)
- FAST-7b: Severe (speech single word)
- FAST-7c: Severe (non-ambulatory)
- FAST-7d: Severe (cannot sit unsupported)
- FAST-7e: Severe (smile lost)
- FAST-7f: Severe/end-stage (cannot hold head up)

CONTENT:
{text[:1000]}

Return ONLY the applicable FAST stages as a comma-separated list (e.g., "FAST-3,FAST-4" or "FAST-6a,FAST-6b,FAST-6c" or "All").
If uncertain, prefix with "?" (e.g., "?FAST-4,FAST-5").

FAST_STAGES:"""

# ============================================================================
# PDF Downloader
# ============================================================================

class PDFDownloader:
    """Downloads PDFs from configured sources."""

    def __init__(self, config: Dict):
        self.config = config
        self.download_dir = Path(config['output']['download_dir'])
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self, pdfs: List[Dict]) -> List[Path]:
        """Download all PDFs if not already present."""
        downloaded = []

        for pdf_info in pdfs:
            local_path = Path(pdf_info['local_path'])

            if local_path.exists():
                logging.info(f"PDF already downloaded: {local_path}")
                downloaded.append(local_path)
                continue

            # Download
            logging.info(f"Downloading {pdf_info['name']} from {pdf_info['url']}")
            try:
                response = requests.get(pdf_info['url'], timeout=60, stream=True)
                response.raise_for_status()

                # Ensure directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Save PDF
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.info(f"Downloaded to {local_path}")
                downloaded.append(local_path)

            except Exception as e:
                logging.error(f"Failed to download {pdf_info['name']}: {e}")

        return downloaded

# ============================================================================
# PDF Text Extractor
# ============================================================================

class PDFTextExtractor:
    """Extracts structured text from PDFs."""

    def __init__(self, library: str = PDF_LIBRARY):
        self.library = library

    def extract(self, pdf_path: Path) -> List[Dict]:
        """Extract text with structure from PDF.

        Returns list of sections: [
            {
                'page': int,
                'text': str,
                'type': 'section' or 'paragraph',
                'level': int (heading level, if applicable)
            }
        ]
        """
        if self.library == 'pdfplumber':
            return self._extract_pdfplumber(pdf_path)
        else:
            return self._extract_pypdf2(pdf_path)

    def _extract_pdfplumber(self, pdf_path: Path) -> List[Dict]:
        """Extract using pdfplumber (better structure preservation)."""
        sections = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                # Split into paragraphs
                paragraphs = text.split('\n\n')

                for para in paragraphs:
                    para = para.strip()
                    if len(para) < 20:  # Skip very short paragraphs
                        continue

                    # Detect if this is likely a heading (all caps, short, ends without period)
                    is_heading = (
                        len(para) < 100 and
                        para.isupper() or
                        (len(para.split()) <= 10 and not para.endswith('.'))
                    )

                    sections.append({
                        'page': page_num,
                        'text': para,
                        'type': 'heading' if is_heading else 'paragraph',
                        'level': 1 if is_heading else 0
                    })

        return sections

    def _extract_pypdf2(self, pdf_path: Path) -> List[Dict]:
        """Extract using PyPDF2 (fallback)."""
        sections = []

        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                # Simple paragraph splitting
                paragraphs = text.split('\n\n')

                for para in paragraphs:
                    para = para.strip()
                    if len(para) < 20:
                        continue

                    sections.append({
                        'page': page_num,
                        'text': para,
                        'type': 'paragraph',
                        'level': 0
                    })

        return sections

# ============================================================================
# Information Item Extractor
# ============================================================================

class InformationExtractor:
    """Extracts information items from PDF sections."""

    def __init__(self, config: Dict, llm: Optional[LLMProvider] = None):
        self.config = config
        self.llm = llm
        self.extract_sections = config['extraction']['extract_sections']
        self.extract_tips = config['extraction']['extract_detailed_tips']

    def extract_from_pdf(self, pdf_info: Dict, sections: List[Dict]) -> List[Dict]:
        """Extract information items from PDF sections."""
        items = []
        current_section = None
        section_buffer = []

        for section in sections:
            # If this is a heading, process previous section first
            if section['type'] == 'heading':
                if current_section and section_buffer:
                    # Extract from buffered section
                    section_items = self._process_section(
                        current_section,
                        section_buffer,
                        pdf_info
                    )
                    items.extend(section_items)

                # Start new section
                current_section = section['text']
                section_buffer = []
            else:
                # Add paragraph to current section buffer
                section_buffer.append(section['text'])

        # Process final section
        if current_section and section_buffer:
            section_items = self._process_section(
                current_section,
                section_buffer,
                pdf_info
            )
            items.extend(section_items)

        return items

    def _process_section(self, heading: str, paragraphs: List[str], pdf_info: Dict) -> List[Dict]:
        """Process a section to extract information items."""
        items = []
        section_text = '\n\n'.join(paragraphs)

        # 1. Extract section-level item (if enabled)
        if self.extract_sections and len(section_text) >= self.config['extraction']['min_section_length']:
            section_item = self._create_section_item(
                heading,
                section_text,
                pdf_info
            )
            if section_item:
                items.append(section_item)

        # 2. Extract detailed tips (if enabled)
        if self.extract_tips:
            tip_items = self._extract_tips_from_section(
                heading,
                section_text,
                pdf_info
            )
            items.extend(tip_items)

        return items

    def _create_section_item(self, heading: str, text: str, pdf_info: Dict) -> Optional[Dict]:
        """Create a section-level information item."""
        # Summarize section (take first 200 chars or use LLM)
        description = text[:200].strip() + "..." if len(text) > 200 else text.strip()

        # Infer metadata
        fast_stages = self._infer_fast_stages(heading + " " + text)
        capability = self._infer_capability(heading + " " + text)
        topics = self._infer_topics(heading + " " + text)

        item = {
            'title': heading,
            'description': description,
            'content_type': 'section',
            'fast_stages': fast_stages,
            'capability': capability,
            'audience': pdf_info['audience'],
            'topics': topics,
            'source_id': pdf_info['id'],
            'source_name': pdf_info['name'],
            'source_type': 'PDF',
            'extraction_method': 'section_level',
            'extraction_date': datetime.now().isoformat(),
            'uncertain_mapping': '?' in fast_stages
        }

        return item

    def _extract_tips_from_section(self, heading: str, text: str, pdf_info: Dict) -> List[Dict]:
        """Extract detailed tips from section text."""
        items = []

        # Look for bullet points, numbered lists, or "tip" patterns
        # Patterns: "•", "-", "1.", "Tip:", etc.

        # Split by common list markers
        lines = text.split('\n')
        current_tip = []

        for line in lines:
            line = line.strip()

            # Check if line starts a new tip
            is_list_item = (
                line.startswith('•') or
                line.startswith('-') or
                re.match(r'^\d+\.', line) or
                line.lower().startswith('tip:') or
                line.lower().startswith('try:')
            )

            if is_list_item:
                # Save previous tip
                if current_tip:
                    tip_text = ' '.join(current_tip).strip()
                    if len(tip_text) >= self.config['extraction']['min_tip_length']:
                        tip_item = self._create_tip_item(
                            heading,
                            tip_text,
                            pdf_info
                        )
                        if tip_item:
                            items.append(tip_item)

                # Start new tip
                current_tip = [line]
            elif current_tip:
                # Continue current tip
                current_tip.append(line)

        # Don't forget last tip
        if current_tip:
            tip_text = ' '.join(current_tip).strip()
            if len(tip_text) >= self.config['extraction']['min_tip_length']:
                tip_item = self._create_tip_item(heading, tip_text, pdf_info)
                if tip_item:
                    items.append(tip_item)

        return items

    def _create_tip_item(self, section_heading: str, tip_text: str, pdf_info: Dict) -> Optional[Dict]:
        """Create an information item from a tip."""
        # Clean up tip text (remove bullet markers, etc.)
        tip_text = re.sub(r'^[•\-\d\.]+\s*', '', tip_text)
        tip_text = re.sub(r'^(Tip|Try):\s*', '', tip_text, flags=re.IGNORECASE)

        # Create title from first few words
        title_words = tip_text.split()[:8]
        title = ' '.join(title_words)
        if len(tip_text.split()) > 8:
            title += '...'

        # Infer metadata
        fast_stages = self._infer_fast_stages(section_heading + " " + tip_text)
        capability = self._infer_capability(section_heading + " " + tip_text)
        topics = self._infer_topics(section_heading + " " + tip_text)

        item = {
            'title': title,
            'description': tip_text,
            'content_type': 'tip',
            'fast_stages': fast_stages,
            'capability': capability,
            'audience': pdf_info['audience'],
            'topics': topics,
            'source_id': pdf_info['id'],
            'source_name': pdf_info['name'],
            'source_type': 'PDF',
            'section': section_heading,
            'extraction_method': 'tip_level',
            'extraction_date': datetime.now().isoformat(),
            'uncertain_mapping': '?' in fast_stages
        }

        return item

    def _infer_fast_stages(self, text: str) -> str:
        """Infer FAST stages from text (rule-based or LLM)."""
        if self.llm and self.llm.client:
            # Use LLM for classification
            result = self.llm.analyze_content(text, prompt_type="classify_stage")
            if result:
                stages = result.strip().replace('FAST_STAGES:', '').strip()
                return stages if stages else "?All"

        # Fallback: rule-based inference
        text_lower = text.lower()

        # Stage keywords
        if any(kw in text_lower for kw in ['early', 'early stage', 'mild cognitive', 'beginning']):
            return 'FAST-3,FAST-4'
        elif any(kw in text_lower for kw in ['moderate', 'middle stage']):
            return 'FAST-4,FAST-5'
        elif any(kw in text_lower for kw in ['later stage', 'advanced', 'severe']):
            return 'FAST-6a,FAST-6b,FAST-6c,FAST-6d,FAST-6e,FAST-7a'
        elif any(kw in text_lower for kw in ['end stage', 'terminal', 'palliative', 'end of life']):
            return 'FAST-7a,FAST-7b,FAST-7c,FAST-7d,FAST-7e,FAST-7f'

        # Mark as uncertain if no clear keywords
        return '?All'

    def _infer_capability(self, text: str) -> str:
        """Infer ADL/IADL capability from text."""
        text_lower = text.lower()

        capability_keywords = {
            'Bathing': ['bath', 'shower', 'wash', 'hygiene'],
            'Dressing': ['dress', 'clothing', 'clothes', 'wear'],
            'Toileting': ['toilet', 'bathroom', 'incontinence'],
            'Transferring': ['transfer', 'mobility', 'walking', 'moving', 'wheelchair'],
            'Continence': ['incontinence', 'bladder', 'bowel', 'continent'],
            'Feeding': ['eating', 'feeding', 'nutrition', 'swallowing', 'meal'],
            'Telephone': ['phone', 'telephone', 'calling'],
            'Shopping': ['shopping', 'groceries', 'store'],
            'Food Preparation': ['cooking', 'meal prep', 'kitchen', 'preparing food'],
            'Housekeeping': ['housekeeping', 'cleaning', 'chores', 'tidy'],
            'Laundry': ['laundry', 'washing clothes'],
            'Transportation': ['driving', 'transportation', 'travel', 'car'],
            'Medications': ['medication', 'medicine', 'pills', 'drugs', 'prescription'],
            'Financial Management': ['financial', 'money', 'bills', 'banking', 'finances', 'budget']
        }

        for capability, keywords in capability_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return capability

        return 'General'

    def _infer_topics(self, text: str) -> str:
        """Infer topics from text."""
        text_lower = text.lower()
        topics = []

        topic_keywords = {
            'Safety': ['safety', 'safe', 'danger', 'risk', 'hazard'],
            'Communication': ['communication', 'talking', 'speaking', 'language'],
            'Behavioral Symptoms': ['behavior', 'behaviour', 'agitation', 'aggression', 'wandering'],
            'Medication Management': ['medication', 'medicine', 'pills'],
            'Legal Planning': ['legal', 'power of attorney', 'advance directive'],
            'Financial Management': ['financial', 'money', 'bills', 'fraud'],
            'Caregiver Support': ['caregiver', 'carer', 'respite', 'stress', 'support'],
            'Activities': ['activities', 'hobbies', 'engagement'],
            'Nutrition': ['nutrition', 'eating', 'diet', 'food', 'hydration'],
            'Personal Care': ['personal care', 'bathing', 'dressing', 'hygiene'],
            'Diagnosis': ['diagnosis', 'screening', 'assessment'],
            'Treatment': ['treatment', 'therapy', 'medication'],
            'End-of-Life Care': ['end of life', 'palliative', 'hospice'],
            'Daily Living': ['daily living', 'routine', 'activities of daily living']
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        return ', '.join(topics[:3]) if topics else 'General Information'

# ============================================================================
# Main Pipeline
# ============================================================================

class PDFExtractionPipeline:
    """Main extraction pipeline for PDF processing."""

    def __init__(self, config_path: str = "config_pdf_extraction.json"):
        self.config = self._load_config(config_path)
        self._setup_logging()

        self.downloader = PDFDownloader(self.config)
        self.pdf_extractor = PDFTextExtractor()

        # Initialize LLM (if configured)
        self.llm = None
        if self.config['llm']['provider'] != 'none':
            self.llm = LLMProvider(self.config)

        self.info_extractor = InformationExtractor(self.config, self.llm)

        self.extracted_items = []

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _setup_logging(self):
        """Set up logging."""
        log_level = getattr(logging, self.config['logging']['level'])
        log_file = self.config['logging']['log_file']

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def run(self):
        """Run extraction pipeline."""
        logging.info("=" * 80)
        logging.info("PDF EXTRACTION PIPELINE STARTED")
        logging.info("=" * 80)

        pdfs = self.config['resources']['pdfs']

        # Download PDFs
        logging.info("Step 1: Downloading PDFs...")
        pdf_paths = self.downloader.download_all(pdfs)

        # Extract from each PDF
        logging.info(f"Step 2: Extracting information from {len(pdf_paths)} PDFs...")

        for pdf_path, pdf_info in zip(pdf_paths, pdfs):
            logging.info(f"\nProcessing: {pdf_info['name']}")

            # Extract text with structure
            logging.info("  Extracting text from PDF...")
            sections = self.pdf_extractor.extract(pdf_path)
            logging.info(f"  Extracted {len(sections)} sections/paragraphs")

            # Extract information items
            logging.info("  Extracting information items...")
            items = self.info_extractor.extract_from_pdf(pdf_info, sections)
            logging.info(f"  Extracted {len(items)} information items")

            self.extracted_items.extend(items)

        # Save results
        self._save_results()

        logging.info("=" * 80)
        logging.info(f"EXTRACTION COMPLETE: {len(self.extracted_items)} total items")
        logging.info("=" * 80)

    def _save_results(self):
        """Save extracted items to CSV."""
        if not self.extracted_items:
            logging.warning("No items to save")
            return

        output_file = self.config['output']['pdf_items']
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.extracted_items)
        df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(df)} items to {output_file}")

        # Summary statistics
        print("\n" + "=" * 80)
        print("EXTRACTION SUMMARY")
        print("=" * 80)
        print(f"Total items: {len(df)}")
        print(f"  - Sections: {len(df[df['content_type'] == 'section'])}")
        print(f"  - Tips: {len(df[df['content_type'] == 'tip'])}")
        print(f"\nBy Source:")
        print(df['source_name'].value_counts())
        print(f"\nBy Audience:")
        print(df['audience'].value_counts())
        print(f"\nUncertain FAST mappings: {len(df[df['uncertain_mapping'] == True])}")
        print("=" * 80)

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("=" * 80)
    print("PDF INFORMATION EXTRACTION")
    print("Health Knowledge Recommender - WP-03")
    print("=" * 80)
    print()

    pipeline = PDFExtractionPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
