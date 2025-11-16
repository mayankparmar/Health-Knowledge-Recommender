#!/usr/bin/env python3
"""
PDF Data Extraction and Annotation System
Health Knowledge Recommender Project

A modular, configurable system for extracting and annotating information
from dementia care PDFs based on FAST staging and ADL/IADL mappings.

Features:
- Multiple PDF extraction libraries (pdfplumber, PyPDF2)
- Rule-based and LLM-based annotation
- Multi-provider LLM support (Claude, GPT, Gemini)
- Knowledge graph output in JSON-LD format
- CSV export for analysis
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import our modules
from src.pdf_extractor.models import PDFDocument, AnnotatedContent
from src.pdf_extractor.loaders import FASTLoader, CapabilityLoader, MappingLoader
from src.pdf_extractor.extractors import PDFExtractor
from src.pdf_extractor.annotators import RuleBasedAnnotator, LLMAnnotator
from src.pdf_extractor.annotators.llm_annotator import create_llm_annotator_from_config
from src.pdf_extractor.builders import KnowledgeGraphBuilder


# Configure logging
def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class ExtractionPipeline:
    """Main extraction and annotation pipeline."""

    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._validate_config()

        # Setup logging
        log_config = self.config.get('logging', {})
        setup_logging(
            log_level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file')
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("PDF EXTRACTION AND ANNOTATION PIPELINE")
        self.logger.info("=" * 80)

        # Initialize components
        self._initialize_loaders()
        self._initialize_extractor()
        self._initialize_annotator()
        self._initialize_knowledge_graph_builder()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate configuration has required fields."""
        required_fields = ['data_paths', 'documents', 'extraction', 'annotation', 'output']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

    def _initialize_loaders(self):
        """Initialize data loaders for FAST, ADL, IADL, and mappings."""
        self.logger.info("Loading reference data...")

        data_paths = self.config['data_paths']

        self.fast_loader = FASTLoader(data_paths['fast_stages'])
        self.capability_loader = CapabilityLoader(
            data_paths['adl_file'],
            data_paths['iadl_file']
        )
        self.mapping_loader = MappingLoader(data_paths['mapping_file'])

        self.logger.info(f"Loaded {len(self.fast_loader.get_all_stages())} FAST stages")
        self.logger.info(f"Loaded {len(self.capability_loader.get_all_capabilities())} capabilities")
        self.logger.info(f"Loaded {len(self.mapping_loader.get_all_mappings())} mappings")

    def _initialize_extractor(self):
        """Initialize PDF extractor."""
        extraction_config = self.config['extraction']

        self.extractor = PDFExtractor(
            library=extraction_config.get('pdf_library', 'auto'),
            min_section_length=extraction_config.get('min_section_length', 50),
            min_paragraph_length=extraction_config.get('min_paragraph_length', 20),
            extract_tips=extraction_config.get('extract_tips', True)
        )

        self.logger.info(f"Initialized PDF extractor with library: {self.extractor.library}")

    def _initialize_annotator(self):
        """Initialize annotator (rule-based or LLM-based)."""
        annotation_config = self.config['annotation']
        method = annotation_config.get('method', 'rule_based')

        if method == 'llm':
            # Try to create LLM annotator
            self.annotator = create_llm_annotator_from_config(
                self.config,
                self.fast_loader,
                self.capability_loader,
                self.mapping_loader
            )

            if self.annotator:
                self.logger.info(f"Initialized LLM annotator: {self.config['llm']['provider']}")
            else:
                self.logger.warning("Failed to initialize LLM annotator, falling back to rule-based")
                self.annotator = self._create_rule_based_annotator()

        elif method == 'hybrid':
            # Create both and use LLM as primary, rule-based as fallback
            self.llm_annotator = create_llm_annotator_from_config(
                self.config,
                self.fast_loader,
                self.capability_loader,
                self.mapping_loader
            )
            self.rule_annotator = self._create_rule_based_annotator()

            if self.llm_annotator:
                self.logger.info("Initialized hybrid annotator (LLM + rule-based fallback)")
                self.annotator = self.llm_annotator
            else:
                self.logger.warning("LLM not available, using rule-based only")
                self.annotator = self.rule_annotator

        else:  # rule_based
            self.annotator = self._create_rule_based_annotator()
            self.logger.info("Initialized rule-based annotator")

    def _create_rule_based_annotator(self) -> RuleBasedAnnotator:
        """Create rule-based annotator."""
        return RuleBasedAnnotator(
            fast_loader=self.fast_loader,
            capability_loader=self.capability_loader,
            mapping_loader=self.mapping_loader,
            min_confidence=0.3
        )

    def _initialize_knowledge_graph_builder(self):
        """Initialize knowledge graph builder."""
        self.kg_builder = KnowledgeGraphBuilder(
            fast_loader=self.fast_loader,
            capability_loader=self.capability_loader,
            mapping_loader=self.mapping_loader,
            base_uri=self.config.get('output', {}).get(
                'knowledge_graph_base_uri',
                'http://health-knowledge-recommender.org/kg/'
            )
        )
        self.logger.info("Initialized knowledge graph builder")

    def run(self):
        """Run the extraction and annotation pipeline."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING PIPELINE")
        self.logger.info("=" * 80)

        # Prepare document list
        documents = self._prepare_documents()

        # Extract and annotate
        all_annotated_content = []

        for doc in documents:
            self.logger.info(f"\nProcessing document: {doc.name}")

            # Extract content
            self.logger.info("  Extracting content...")
            extracted_contents = self.extractor.extract(doc)
            self.logger.info(f"  Extracted {len(extracted_contents)} content items")

            # Annotate content
            self.logger.info("  Annotating content...")
            annotations = self.annotator.annotate_batch(extracted_contents, doc)
            self.logger.info(f"  Created {len(annotations)} annotations")

            # Combine
            for content, annotation in zip(extracted_contents, annotations):
                all_annotated_content.append(
                    AnnotatedContent(content=content, annotation=annotation)
                )

        self.logger.info(f"\nTotal annotated content items: {len(all_annotated_content)}")

        # Build knowledge graph
        self._build_and_save_knowledge_graph(documents, all_annotated_content)

        # Save outputs
        self._save_outputs(all_annotated_content)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)

    def _prepare_documents(self) -> List[PDFDocument]:
        """Prepare PDFDocument objects from configuration."""
        documents = []

        for doc_config in self.config['documents']:
            doc = PDFDocument(
                doc_id=doc_config['id'],
                name=doc_config['name'],
                file_path=doc_config['file_path'],
                url=doc_config.get('url'),
                source_organization=doc_config.get('source_organization'),
                target_audience=doc_config.get('target_audience'),
                publication_date=doc_config.get('publication_date'),
                language=doc_config.get('language', 'en')
            )

            # Check if file exists
            if not Path(doc.file_path).exists():
                self.logger.warning(f"File not found: {doc.file_path}, skipping")
                continue

            documents.append(doc)

        self.logger.info(f"Prepared {len(documents)} documents for processing")
        return documents

    def _build_and_save_knowledge_graph(
        self,
        documents: List[PDFDocument],
        annotated_contents: List[AnnotatedContent]
    ):
        """Build and save knowledge graph."""
        output_config = self.config['output']

        if not output_config.get('generate_knowledge_graph', True):
            self.logger.info("Knowledge graph generation disabled")
            return

        self.logger.info("\nBuilding knowledge graph...")

        kg = self.kg_builder.build(
            documents=documents,
            annotated_contents=annotated_contents,
            include_reference_data=output_config.get('include_reference_data', True)
        )

        # Save to JSON-LD
        kg_file = output_config.get('knowledge_graph_file', 'output/knowledge_graph.jsonld')
        self.kg_builder.save(kg_file, pretty=True)

        # Print statistics
        stats = self.kg_builder.get_statistics()
        self.logger.info(f"\nKnowledge Graph Statistics:")
        self.logger.info(f"  Total nodes: {stats['total_nodes']}")
        for node_type, count in stats['by_type'].items():
            self.logger.info(f"    {node_type}: {count}")

    def _save_outputs(self, annotated_contents: List[AnnotatedContent]):
        """Save outputs in various formats."""
        output_config = self.config['output']
        output_dir = Path(output_config.get('output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV views
        if output_config.get('generate_csv', True):
            self.logger.info("\nSaving CSV views...")
            self.kg_builder.save_csv_views(str(output_dir))

        # Save raw JSON
        if output_config.get('generate_json', False):
            import json
            json_file = output_dir / 'annotated_content.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [ac.to_dict() for ac in annotated_contents],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            self.logger.info(f"Saved JSON to {json_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and annotate information from dementia care PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python extract_and_annotate.py

  # Run with custom config
  python extract_and_annotate.py --config my_config.yaml

  # Use LLM-based annotation
  python extract_and_annotate.py --annotation-method llm

  # Generate only knowledge graph (no CSV)
  python extract_and_annotate.py --no-csv
        """
    )

    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--annotation-method',
        choices=['rule_based', 'llm', 'hybrid'],
        help='Override annotation method from config'
    )

    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Do not generate CSV outputs'
    )

    parser.add_argument(
        '--no-kg',
        action='store_true',
        help='Do not generate knowledge graph'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    try:
        # Load and modify config based on arguments
        pipeline = ExtractionPipeline(args.config)

        # Override config with command-line arguments
        if args.annotation_method:
            pipeline.config['annotation']['method'] = args.annotation_method
            # Re-initialize annotator
            pipeline._initialize_annotator()

        if args.no_csv:
            pipeline.config['output']['generate_csv'] = False

        if args.no_kg:
            pipeline.config['output']['generate_knowledge_graph'] = False

        # Run pipeline
        pipeline.run()

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
