"""
Loader for FAST (Functional Assessment Staging Tool) data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from ..models import FASTStage

logger = logging.getLogger(__name__)


class FASTLoader:
    """Loads FAST stage data from JSON file."""

    def __init__(self, fast_file_path: str):
        """
        Initialize FAST loader.

        Args:
            fast_file_path: Path to fast-stages.json file
        """
        self.fast_file_path = Path(fast_file_path)
        self.stages: Dict[str, FASTStage] = {}
        self._load()

    def _load(self):
        """Load FAST stages from JSON file."""
        try:
            with open(self.fast_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stages_data = data.get('stages', [])
            for stage_info in stages_data:
                functional_status = stage_info.get('functional_status', {})

                stage = FASTStage(
                    stage_code=stage_info['stage_code'],
                    stage_name=stage_info['stage_name'],
                    clinical_characteristics=stage_info['clinical_characteristics'],
                    cognition=functional_status.get('cognition', ''),
                    adl_status=functional_status.get('adl_status', ''),
                    iadl_status=functional_status.get('iadl_status', ''),
                    typical_duration=stage_info.get('typical_duration'),
                    care_needs=stage_info.get('care_needs')
                )

                self.stages[stage.stage_code] = stage

            logger.info(f"Loaded {len(self.stages)} FAST stages from {self.fast_file_path}")

        except Exception as e:
            logger.error(f"Failed to load FAST stages: {e}")
            raise

    def get_stage(self, stage_code: str) -> FASTStage:
        """
        Get FAST stage by code.

        Args:
            stage_code: Stage code (e.g., "FAST-3", "FAST-6a")

        Returns:
            FASTStage object

        Raises:
            KeyError: If stage code not found
        """
        return self.stages[stage_code]

    def get_all_stages(self) -> List[FASTStage]:
        """
        Get all FAST stages ordered by stage number.

        Returns:
            List of FASTStage objects
        """
        return sorted(self.stages.values(), key=lambda s: s.stage_number)

    def get_stage_codes(self) -> List[str]:
        """Get all stage codes."""
        return list(self.stages.keys())

    def get_stages_in_range(self, start_stage: str, end_stage: str) -> List[FASTStage]:
        """
        Get all stages within a range (inclusive).

        Args:
            start_stage: Starting stage code (e.g., "FAST-3")
            end_stage: Ending stage code (e.g., "FAST-5")

        Returns:
            List of FASTStage objects in range
        """
        start_num = self.get_stage(start_stage).stage_number
        end_num = self.get_stage(end_stage).stage_number

        return [
            stage for stage in self.get_all_stages()
            if start_num <= stage.stage_number <= end_num
        ]
