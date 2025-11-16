"""
Loader for ADL and IADL capability data.
"""

import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd

from ..models import Capability, CapabilityType

logger = logging.getLogger(__name__)


class CapabilityLoader:
    """Loads ADL and IADL capability data from Excel files."""

    def __init__(self, adl_file_path: str, iadl_file_path: str):
        """
        Initialize capability loader.

        Args:
            adl_file_path: Path to ADL Excel file
            iadl_file_path: Path to IADL Excel file
        """
        self.adl_file_path = Path(adl_file_path)
        self.iadl_file_path = Path(iadl_file_path)
        self.capabilities: Dict[str, Capability] = {}
        self._load()

    def _load(self):
        """Load capabilities from Excel files."""
        try:
            # Load ADLs
            adl_df = pd.read_excel(self.adl_file_path)
            for _, row in adl_df.iterrows():
                capability = Capability(
                    capability_id=row['adl_id'],
                    capability_type=CapabilityType.ADL,
                    name=row['adl_name'],
                    description=row['description'],
                    independence_criteria=row.get('independence_criteria'),
                    dependence_indicators=row.get('dependence_indicators'),
                    icf_code_primary=row.get('icf_code_primary'),
                    icf_code_secondary=row.get('icf_code_secondary'),
                    icf_category=row.get('icf_category')
                )
                self.capabilities[capability.capability_id] = capability

            # Load IADLs
            iadl_df = pd.read_excel(self.iadl_file_path)
            for _, row in iadl_df.iterrows():
                capability = Capability(
                    capability_id=row['iadl_id'],
                    capability_type=CapabilityType.IADL,
                    name=row['iadl_name'],
                    description=row['description'],
                    icf_code_primary=row.get('icf_code_primary'),
                    icf_code_secondary=row.get('icf_code_secondary'),
                    icf_category=row.get('icf_category')
                )
                self.capabilities[capability.capability_id] = capability

            logger.info(f"Loaded {len(self.capabilities)} capabilities "
                       f"({len(adl_df)} ADLs, {len(iadl_df)} IADLs)")

        except Exception as e:
            logger.error(f"Failed to load capabilities: {e}")
            raise

    def get_capability(self, capability_id: str) -> Capability:
        """
        Get capability by ID.

        Args:
            capability_id: Capability ID (e.g., "ADL-1", "IADL-3")

        Returns:
            Capability object

        Raises:
            KeyError: If capability ID not found
        """
        return self.capabilities[capability_id]

    def get_all_capabilities(self) -> List[Capability]:
        """Get all capabilities."""
        return list(self.capabilities.values())

    def get_capabilities_by_type(self, capability_type: CapabilityType) -> List[Capability]:
        """
        Get capabilities by type.

        Args:
            capability_type: Type of capability (ADL or IADL)

        Returns:
            List of Capability objects
        """
        return [
            cap for cap in self.capabilities.values()
            if cap.capability_type == capability_type
        ]

    def get_capability_ids(self) -> List[str]:
        """Get all capability IDs."""
        return list(self.capabilities.keys())

    def search_by_keyword(self, keyword: str) -> List[Capability]:
        """
        Search capabilities by keyword in name or description.

        Args:
            keyword: Keyword to search for (case-insensitive)

        Returns:
            List of matching Capability objects
        """
        keyword_lower = keyword.lower()
        return [
            cap for cap in self.capabilities.values()
            if keyword_lower in cap.name.lower() or
               keyword_lower in cap.description.lower()
        ]
