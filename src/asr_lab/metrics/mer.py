import jiwer
from typing import Any, Dict, Optional

from .base import Metric


class MER(Metric):
    """
    Calculates the Match Error Rate (MER) using the jiwer library.
    
    MER is similar to WER but normalizes differently. It measures the ratio of
    errors to the total number of words in the alignment (reference + hypothesis
    words that were aligned).
    
    Formula: MER = (S + D + I) / (S + D + C)
    Where:
        S = number of word substitutions
        D = number of word deletions
        I = number of word insertions
        C = number of correct words
        
    MER is always between 0 and 1, unlike WER which can exceed 1.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def compute(self, prediction: str, reference: str, **kwargs) -> float:
        """
        Computes the MER between a prediction and a reference transcription.
        
        Args:
            prediction: The transcribed text from the ASR engine.
            reference: The ground truth transcription.
            
        Returns:
            The Match Error Rate as a float between 0 and 1.
        """
        mer_value = jiwer.mer(reference, prediction)
        return mer_value

    def get_display_name(self) -> str:
        """Returns the display name of the metric."""
        return "Match Error Rate (MER)"

    def is_lower_better(self) -> bool:
        """Returns True, as a lower MER is better."""
        return True
