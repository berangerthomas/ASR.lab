from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class Metric(ABC):
    """Abstract base class for all metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Use the name from the config if provided, otherwise default to the class name
        self.name = self.config.get("name", self.__class__.__name__)
    
    @abstractmethod
    def compute(self, prediction: str, reference: str, **kwargs) -> Any:
        """
        Computes the metric.
        Can return a single float or a dictionary with detailed results.
        """
        pass
    
    @abstractmethod
    def get_display_name(self) -> str:
        """Returns the display name of the metric"""
        pass
    
    @abstractmethod
    def is_lower_better(self) -> bool:
        """Returns True if a lower primary metric value is better"""
        pass
