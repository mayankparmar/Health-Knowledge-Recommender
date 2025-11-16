"""
Loader for FAST-ADL-IADL mapping data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from ..models import FASTCapabilityMapping, CapabilityType, ImpairmentLevel

logger = logging.getLogger(__name__)


class MappingLoader:
    """Loads FAST to ADL/IADL mapping data from Excel file."""

    def __init__(self, mapping_file_path: str):
        """
        Initialize mapping loader.

        Args:
            mapping_file_path: Path to mapping Excel file
        """
        self.mapping_file_path = Path(mapping_file_path)
        self.mappings: List[FASTCapabilityMapping] = []
        self.mappings_by_fast: Dict[str, List[FASTCapabilityMapping]] = {}
        self.mappings_by_capability: Dict[str, List[FASTCapabilityMapping]] = {}
        self._load()

    def _load(self):
        """Load mappings from Excel file."""
        try:
            df = pd.read_excel(self.mapping_file_path)

            # Filter out rows with NaN in critical columns
            df = df.dropna(subset=['fast_stage', 'capability_id', 'impairment_level'])

            for _, row in df.iterrows():
                # Parse capability type
                cap_type = CapabilityType.ADL if row['capability_type'] == 'ADL' else CapabilityType.IADL

                # Parse impairment level
                try:
                    impairment = ImpairmentLevel(row['impairment_level'])
                except ValueError:
                    logger.warning(f"Unknown impairment level: {row['impairment_level']}, skipping")
                    continue

                mapping = FASTCapabilityMapping(
                    fast_stage=row['fast_stage'],
                    capability_id=row['capability_id'],
                    capability_name=row['capability_name'],
                    capability_type=cap_type,
                    impairment_level=impairment,
                    clinical_notes=row.get('clinical_notes'),
                    information_needs=row.get('information_needs'),
                    icf_code=row.get('icf_code')
                )

                self.mappings.append(mapping)

                # Index by FAST stage
                if mapping.fast_stage not in self.mappings_by_fast:
                    self.mappings_by_fast[mapping.fast_stage] = []
                self.mappings_by_fast[mapping.fast_stage].append(mapping)

                # Index by capability
                if mapping.capability_id not in self.mappings_by_capability:
                    self.mappings_by_capability[mapping.capability_id] = []
                self.mappings_by_capability[mapping.capability_id].append(mapping)

            logger.info(f"Loaded {len(self.mappings)} FAST-Capability mappings from {self.mapping_file_path}")

        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
            raise

    def get_all_mappings(self) -> List[FASTCapabilityMapping]:
        """Get all mappings."""
        return self.mappings

    def get_mappings_for_fast_stage(self, fast_stage: str) -> List[FASTCapabilityMapping]:
        """
        Get all capability mappings for a FAST stage.

        Args:
            fast_stage: FAST stage code (e.g., "FAST-3")

        Returns:
            List of FASTCapabilityMapping objects
        """
        return self.mappings_by_fast.get(fast_stage, [])

    def get_mappings_for_capability(self, capability_id: str) -> List[FASTCapabilityMapping]:
        """
        Get all FAST stage mappings for a capability.

        Args:
            capability_id: Capability ID (e.g., "ADL-1")

        Returns:
            List of FASTCapabilityMapping objects
        """
        return self.mappings_by_capability.get(capability_id, [])

    def get_impaired_capabilities_for_stage(
        self,
        fast_stage: str,
        min_impairment: ImpairmentLevel = ImpairmentLevel.REQUIRES_PROMPTING
    ) -> List[str]:
        """
        Get list of impaired capability IDs for a FAST stage.

        Args:
            fast_stage: FAST stage code
            min_impairment: Minimum impairment level to consider

        Returns:
            List of capability IDs
        """
        # Define impairment hierarchy
        impairment_order = {
            ImpairmentLevel.INDEPENDENT: 0,
            ImpairmentLevel.REQUIRES_PROMPTING: 1,
            ImpairmentLevel.IMPAIRED: 2,
            ImpairmentLevel.REQUIRES_ASSISTANCE: 3,
            ImpairmentLevel.DEPENDENT: 4
        }

        min_level = impairment_order[min_impairment]

        mappings = self.get_mappings_for_fast_stage(fast_stage)
        return [
            m.capability_id for m in mappings
            if impairment_order[m.impairment_level] >= min_level
        ]

    def get_information_needs_for_stage(self, fast_stage: str) -> Dict[str, str]:
        """
        Get information needs for a FAST stage grouped by capability.

        Args:
            fast_stage: FAST stage code

        Returns:
            Dictionary mapping capability_id to information_needs
        """
        mappings = self.get_mappings_for_fast_stage(fast_stage)
        return {
            m.capability_id: m.information_needs
            for m in mappings
            if m.information_needs and pd.notna(m.information_needs)
        }

    def get_stages_for_capability_impairment(
        self,
        capability_id: str,
        impairment_level: ImpairmentLevel
    ) -> List[str]:
        """
        Get FAST stages where a capability has a specific impairment level.

        Args:
            capability_id: Capability ID
            impairment_level: Desired impairment level

        Returns:
            List of FAST stage codes
        """
        mappings = self.get_mappings_for_capability(capability_id)
        return [
            m.fast_stage for m in mappings
            if m.impairment_level == impairment_level
        ]
