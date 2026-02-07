import jiwer
from typing import Any, Dict, List, Optional

from .base import Metric


def create_text_transforms(config: Dict[str, Any], for_chars: bool = False) -> jiwer.Compose:
    """
    Creates a jiwer text transformation pipeline based on config options.
    
    Available config options:
        - lowercase (bool): Convert to lowercase (default: True)
        - remove_punctuation (bool): Remove punctuation (default: True)
        - expand_contractions (bool): Expand English contractions (default: False)
    
    Args:
        config: Configuration dictionary
        for_chars: If True, reduces to characters (for CER), otherwise to words (for WER)
    """
    transforms: List[jiwer.AbstractTransform] = []
    
    if config.get("lowercase", True):
        transforms.append(jiwer.ToLowerCase())
    
    if config.get("remove_punctuation", True):
        transforms.append(jiwer.RemovePunctuation())
    
    if config.get("expand_contractions", False):
        transforms.append(jiwer.ExpandCommonEnglishContractions())
    
    # Always normalize spacing for consistent comparison
    transforms.extend([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    # Required by jiwer: reduce to list of words or characters
    if for_chars:
        transforms.append(jiwer.ReduceToListOfListOfChars())
    else:
        transforms.append(jiwer.ReduceToListOfListOfWords())
    
    return jiwer.Compose(transforms)


class WER(Metric):
    """
    Calculates the Word Error Rate (WER) with optional text normalization.
    
    Text normalization options (via config):
        - lowercase: true/false (default: true)
        - remove_punctuation: true/false (default: true)
        - expand_contractions: true/false (default: false)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.transforms = create_text_transforms(self.config, for_chars=False)

    def compute(self, prediction: str, reference: str, **kwargs) -> float:
        """
        Computes the WER between a prediction and a reference transcription.
        Both texts are normalized using the configured transforms before comparison.
        """
        wer_value = jiwer.wer(
            reference=reference,
            hypothesis=prediction,
            reference_transform=self.transforms,
            hypothesis_transform=self.transforms,
        )
        return wer_value

    def get_display_name(self) -> str:
        """Returns the display name of the metric."""
        return "Word Error Rate (WER)"

    def is_lower_better(self) -> bool:
        """Returns True, as a lower WER is better."""
        return True
