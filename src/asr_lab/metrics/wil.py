import jiwer
from typing import Any, Dict, Optional

from .base import Metric


class WIL(Metric):
    """
    Calculates the Word Information Lost (WIL) metric.
    
    WIL measures the proportion of word information lost during transcription.
    It considers both precision and recall aspects of the transcription.
    
    Formula: WIL = 1 - (C² / (N × M))
    Where:
        C = number of correct words
        N = total words in reference
        M = total words in hypothesis
        
    WIL ranges from 0 (perfect) to 1 (complete mismatch).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def compute(self, prediction: str, reference: str, **kwargs) -> float:
        """
        Computes the WIL between a prediction and a reference transcription.
        
        Args:
            prediction: The transcribed text from the ASR engine.
            reference: The ground truth transcription.
            
        Returns:
            The Word Information Lost as a float between 0 and 1.
        """
        wil_value = jiwer.wil(reference, prediction)
        return wil_value

    def get_display_name(self) -> str:
        """Returns the display name of the metric."""
        return "Word Information Lost (WIL)"

    def is_lower_better(self) -> bool:
        """Returns True, as a lower WIL is better."""
        return True
