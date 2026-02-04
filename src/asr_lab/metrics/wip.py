import jiwer
from typing import Any, Dict, Optional

from .base import Metric


class WIP(Metric):
    """
    Calculates the Word Information Preserved (WIP) metric.
    
    WIP is the complement of WIL and measures the proportion of word information
    preserved during transcription. Higher is better.
    
    Formula: WIP = C² / (N × M)
    Where:
        C = number of correct words
        N = total words in reference
        M = total words in hypothesis
        
    WIP ranges from 0 (complete mismatch) to 1 (perfect transcription).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def compute(self, prediction: str, reference: str, **kwargs) -> float:
        """
        Computes the WIP between a prediction and a reference transcription.
        
        Args:
            prediction: The transcribed text from the ASR engine.
            reference: The ground truth transcription.
            
        Returns:
            The Word Information Preserved as a float between 0 and 1.
        """
        wip_value = jiwer.wip(reference, prediction)
        return wip_value

    def get_display_name(self) -> str:
        """Returns the display name of the metric."""
        return "Word Information Preserved (WIP)"

    def is_lower_better(self) -> bool:
        """Returns False, as a higher WIP is better."""
        return False
