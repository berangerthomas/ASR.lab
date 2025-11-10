from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

class ASREngine(ABC):
    """Abstract base class for all ASR engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("id", self.__class__.__name__)
        
    @abstractmethod
    def load_model(self) -> None:
        """Loads the ASR model"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str) -> Dict[str, Any]:
        """
        Transcribes an audio file.

        Returns:
            A dictionary containing transcription details, e.g.,
            {
                'text': str,
                'confidence': float,
                'processing_time': float
            }
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the model"""
        pass
