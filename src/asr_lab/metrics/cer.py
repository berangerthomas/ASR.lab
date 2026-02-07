import jiwer
from typing import Any, Dict, Optional

from .base import Metric
from .wer import create_text_transforms


class CER(Metric):
    """
    Calculates the Character Error Rate (CER) with optional text normalization.
    
    CER measures the edit distance at the character level between the prediction
    and the reference. It's particularly useful for languages without clear word
    boundaries (e.g., Chinese, Japanese) or for evaluating spelling accuracy.
    
    Formula: CER = (S + D + I) / N
    Where:
        S = number of character substitutions
        D = number of character deletions
        I = number of character insertions
        N = total number of characters in the reference
    
    Text normalization options (via config):
        - lowercase: true/false (default: true)
        - remove_punctuation: true/false (default: true)
        - expand_contractions: true/false (default: false)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.transforms = create_text_transforms(self.config, for_chars=True)

    def compute(self, prediction: str, reference: str, **kwargs) -> float:
        """
        Computes the CER between a prediction and a reference transcription.
        Both texts are normalized using the configured transforms before comparison.
        
        Args:
            prediction: The transcribed text from the ASR engine.
            reference: The ground truth transcription.
            
        Returns:
            The Character Error Rate as a float between 0 and potentially > 1
            (if there are many insertions).
        """
        cer_value = jiwer.cer(
            reference=reference,
            hypothesis=prediction,
            reference_transform=self.transforms,
            hypothesis_transform=self.transforms,
        )
        return cer_value

    def get_display_name(self) -> str:
        """Returns the display name of the metric."""
        return "Character Error Rate (CER)"

    def is_lower_better(self) -> bool:
        """Returns True, as a lower CER is better."""
        return True
