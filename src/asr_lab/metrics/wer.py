import evaluate
from typing import Any, Dict, Optional

from .base import Metric

class WER(Metric):
    """
    Calculates the Word Error Rate (WER) using the Hugging Face Evaluate library.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.wer_metric = evaluate.load("wer")

    def compute(self, prediction: str, reference: str, **kwargs) -> float:
        """
        Computes the WER between a prediction and a reference transcription.
        """
        # The compute method expects lists of predictions and references.
        wer_value = self.wer_metric.compute(
            predictions=[prediction],
            references=[reference]
        )
        return wer_value

    def get_display_name(self) -> str:
        """Returns the display name of the metric."""
        return "Word Error Rate (WER)"

    def is_lower_better(self) -> bool:
        """Returns True, as a lower WER is better."""
        return True
