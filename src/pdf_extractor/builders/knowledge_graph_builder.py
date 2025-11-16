"""
Knowledge Graph Builder

Creates a knowledge graph in JSON-LD format from annotated content.
The graph can be imported into Neo4j, RDF stores, or other graph databases.
"""

import logging
import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from ..models import (
    AnnotatedContent, ExtractedContent, Annotation,
    PDFDocument, FASTStage, Capability, FASTCapabilityMapping
)
from ..loaders import FASTLoader, CapabilityLoader, MappingLoader

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph from annotated content.

    Output format: JSON-LD (JSON for Linked Data)
    - Compatible with RDF/OWL ontologies
    - Easy to import into graph databases
    - Supports semantic queries and reasoning
    """

    def __init__(
        self,
        fast_loader: FASTLoader,
        capability_loader: CapabilityLoader,
        mapping_loader: MappingLoader,
        base_uri: str = "http://health-knowledge-recommender.org/kg/"
    ):
        """
        Initialize knowledge graph builder.

        Args:
            fast_loader: FAST stage data loader
            capability_loader: Capability data loader
            mapping_loader: Mapping data loader
            base_uri: Base URI for knowledge graph entities
        """
        self.fast_loader = fast_loader
        self.capability_loader = capability_loader
        self.mapping_loader = mapping_loader
        self.base_uri = base_uri

        self.graph = {
            "@context": self._create_context(),
            "@graph": []
        }

    def _create_context(self) -> Dict[str, Any]:
        """Create JSON-LD context defining vocabularies and prefixes."""
        return {
            "@vocab": self.base_uri,
            "schema": "http://schema.org/",
            "fast": f"{self.base_uri}fast/",
            "capability": f"{self.base_uri}capability/",
            "content": f"{self.base_uri}content/",
            "document": f"{self.base_uri}document/",
            "icf": "http://purl.bioontology.org/ontology/ICF/",

            # Property definitions
            "hasStage": {"@type": "@id"},
            "hasCapability": {"@type": "@id"},
            "appliesTo": {"@type": "@id"},
            "relatedTo": {"@type": "@id"},
            "extractedFrom": {"@type": "@id"},
            "hasImpairment": {"@type": "@id"},
            "confidence": {"@type": "schema:Float"},
            "pageNumber": {"@type": "schema:Integer"}
        }

    def add_documents(self, documents: List[PDFDocument]):
        """
        Add document nodes to the graph.

        Args:
            documents: List of PDFDocument objects
        """
        for doc in documents:
            node = {
                "@id": f"document:{doc.doc_id}",
                "@type": "Document",
                "name": doc.name,
                "filePath": doc.file_path,
                "url": doc.url,
                "sourceOrganization": doc.source_organization,
                "targetAudience": doc.target_audience,
                "publicationDate": doc.publication_date,
                "language": doc.language
            }
            # Remove None values
            node = {k: v for k, v in node.items() if v is not None}
            self.graph["@graph"].append(node)

    def add_fast_stages(self):
        """Add FAST stage nodes to the graph."""
        for stage in self.fast_loader.get_all_stages():
            node = {
                "@id": f"fast:{stage.stage_code}",
                "@type": "FASTStage",
                "stageCode": stage.stage_code,
                "stageName": stage.stage_name,
                "stageNumber": stage.stage_number,
                "clinicalCharacteristics": stage.clinical_characteristics,
                "cognition": stage.cognition,
                "adlStatus": stage.adl_status,
                "iadlStatus": stage.iadl_status,
                "typicalDuration": stage.typical_duration,
                "careNeeds": stage.care_needs
            }
            # Remove None values
            node = {k: v for k, v in node.items() if v is not None}
            self.graph["@graph"].append(node)

    def add_capabilities(self):
        """Add capability (ADL/IADL) nodes to the graph."""
        for cap in self.capability_loader.get_all_capabilities():
            node = {
                "@id": f"capability:{cap.capability_id}",
                "@type": f"{cap.capability_type.value}Capability",
                "capabilityId": cap.capability_id,
                "capabilityType": cap.capability_type.value,
                "name": cap.name,
                "description": cap.description,
                "independenceCriteria": cap.independence_criteria,
                "dependenceIndicators": cap.dependence_indicators,
                "icfCodePrimary": cap.icf_code_primary,
                "icfCodeSecondary": cap.icf_code_secondary,
                "icfCategory": cap.icf_category
            }
            # Remove None values
            node = {k: v for k, v in node.items() if v is not None}
            self.graph["@graph"].append(node)

    def add_mappings(self):
        """Add FAST-Capability mapping edges to the graph."""
        for mapping in self.mapping_loader.get_all_mappings():
            node = {
                "@id": f"mapping:{mapping.fast_stage}_{mapping.capability_id}",
                "@type": "FASTCapabilityMapping",
                "fastStage": {"@id": f"fast:{mapping.fast_stage}"},
                "capability": {"@id": f"capability:{mapping.capability_id}"},
                "impairmentLevel": mapping.impairment_level.value,
                "clinicalNotes": mapping.clinical_notes,
                "informationNeeds": mapping.information_needs,
                "icfCode": mapping.icf_code
            }
            # Remove None values
            node = {k: v for k, v in node.items() if v is not None}
            self.graph["@graph"].append(node)

    def add_annotated_content(
        self,
        annotated_contents: List[AnnotatedContent]
    ):
        """
        Add annotated content nodes and their relationships to the graph.

        Args:
            annotated_contents: List of AnnotatedContent objects
        """
        for ac in annotated_contents:
            content = ac.content
            annotation = ac.annotation

            # Content node
            content_node = {
                "@id": f"content:{content.content_id}",
                "@type": f"{content.content_type.value.capitalize()}Content",
                "contentId": content.content_id,
                "title": content.title,
                "text": content.text,
                "contentType": content.content_type.value,
                "pageNumber": content.page_number,
                "sectionHierarchy": content.section_hierarchy,
                "extractedFrom": {"@id": f"document:{content.doc_id}"},
                "extractionMethod": content.extraction_method.value,
                "extractionTimestamp": content.extraction_timestamp
            }
            # Remove None values
            content_node = {k: v for k, v in content_node.items() if v is not None}
            self.graph["@graph"].append(content_node)

            # Annotation node
            annotation_node = {
                "@id": f"annotation:{annotation.annotation_id}",
                "@type": "Annotation",
                "annotationId": annotation.annotation_id,
                "annotates": {"@id": f"content:{content.content_id}"},
                "fastStages": [{"@id": f"fast:{stage}"} for stage in annotation.fast_stages],
                "fastConfidence": annotation.fast_confidence,
                "capabilities": [{"@id": f"capability:{cap}"} for cap in annotation.capabilities],
                "capabilityConfidence": annotation.capability_confidence,
                "topics": annotation.topics,
                "keywords": annotation.keywords,
                "targetAudience": annotation.target_audience,
                "annotationMethod": annotation.annotation_method.value,
                "annotationTimestamp": annotation.annotation_timestamp,
                "annotatorNotes": annotation.annotator_notes
            }
            # Remove None values
            annotation_node = {k: v for k, v in annotation_node.items() if v is not None}
            self.graph["@graph"].append(annotation_node)

    def build(
        self,
        documents: List[PDFDocument],
        annotated_contents: List[AnnotatedContent],
        include_reference_data: bool = True
    ) -> Dict[str, Any]:
        """
        Build the complete knowledge graph.

        Args:
            documents: List of PDF documents
            annotated_contents: List of annotated content
            include_reference_data: Whether to include FAST/ADL/IADL reference data

        Returns:
            Complete knowledge graph as JSON-LD dictionary
        """
        logger.info("Building knowledge graph...")

        # Reset graph
        self.graph["@graph"] = []

        # Add metadata
        metadata = {
            "@id": f"{self.base_uri}metadata",
            "@type": "KnowledgeGraphMetadata",
            "createdAt": datetime.now().isoformat(),
            "documentCount": len(documents),
            "contentCount": len(annotated_contents),
            "version": "2.0.0"
        }
        self.graph["@graph"].append(metadata)

        # Add reference data (FAST stages, capabilities, mappings)
        if include_reference_data:
            logger.info("Adding FAST stages...")
            self.add_fast_stages()

            logger.info("Adding capabilities...")
            self.add_capabilities()

            logger.info("Adding mappings...")
            self.add_mappings()

        # Add documents
        logger.info("Adding documents...")
        self.add_documents(documents)

        # Add annotated content
        logger.info("Adding annotated content...")
        self.add_annotated_content(annotated_contents)

        logger.info(f"Knowledge graph built with {len(self.graph['@graph'])} nodes")

        return self.graph

    def save(self, output_path: str, pretty: bool = True):
        """
        Save knowledge graph to JSON-LD file.

        Args:
            output_path: Path to save file
            pretty: Whether to pretty-print JSON
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(self.graph, f, indent=2, ensure_ascii=False)
            else:
                json.dump(self.graph, f, ensure_ascii=False)

        logger.info(f"Knowledge graph saved to {output_file}")

    def save_csv_views(self, output_dir: str):
        """
        Save CSV views of the knowledge graph for analysis.

        Args:
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        # Extract nodes by type
        contents = []
        annotations = []
        for node in self.graph["@graph"]:
            node_type = node.get("@type", "")

            if "Content" in node_type:
                contents.append({
                    "content_id": node.get("contentId"),
                    "type": node.get("contentType"),
                    "title": node.get("title"),
                    "text": node.get("text", "")[:200],  # Truncate for CSV
                    "page": node.get("pageNumber"),
                    "doc_id": node.get("extractedFrom", {}).get("@id", "").replace("document:", "")
                })

            elif node_type == "Annotation":
                # Extract stage and capability IDs
                fast_stages = [s.get("@id", "").replace("fast:", "") for s in node.get("fastStages", [])]
                capabilities = [c.get("@id", "").replace("capability:", "") for c in node.get("capabilities", [])]

                annotations.append({
                    "annotation_id": node.get("annotationId"),
                    "content_id": node.get("annotates", {}).get("@id", "").replace("content:", ""),
                    "fast_stages": ",".join(fast_stages),
                    "fast_confidence": node.get("fastConfidence"),
                    "capabilities": ",".join(capabilities),
                    "capability_confidence": node.get("capabilityConfidence"),
                    "topics": ",".join(node.get("topics", [])),
                    "target_audience": node.get("targetAudience"),
                    "method": node.get("annotationMethod")
                })

        # Save to CSV
        if contents:
            df_contents = pd.DataFrame(contents)
            df_contents.to_csv(output_path / "contents.csv", index=False)
            logger.info(f"Saved {len(contents)} contents to {output_path}/contents.csv")

        if annotations:
            df_annotations = pd.DataFrame(annotations)
            df_annotations.to_csv(output_path / "annotations.csv", index=False)
            logger.info(f"Saved {len(annotations)} annotations to {output_path}/annotations.csv")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_nodes": len(self.graph["@graph"]),
            "by_type": {}
        }

        # Count by type
        for node in self.graph["@graph"]:
            node_type = node.get("@type", "Unknown")
            stats["by_type"][node_type] = stats["by_type"].get(node_type, 0) + 1

        return stats
