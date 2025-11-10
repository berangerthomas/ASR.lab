import time
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import HubertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
import librosa

from .base import ASREngine


class HubertEngine(ASREngine):
    """ASR engine for Meta's HuBERT models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_id = self.config.get("model_id")
        if not self.model_id:
            raise ValueError("HubertEngine requires 'model_id'")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Loads HuBERT model and feature extractor."""
        if self.model is None:
            # HuBERT models don't have their own tokenizer, use Wav2Vec2's
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
            
            # Use a compatible tokenizer from wav2vec2-base for decoding
            try:
                self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
            except Exception:
                # Fallback: create simple character-based decoding
                self.tokenizer = None
            
            self.model = HubertForCTC.from_pretrained(self.model_id).to(self.device)
            self.model.eval()

    def transcribe(self, audio_path: Path, language: str) -> Dict[str, Any]:
        """Transcribes audio using HuBERT."""
        if self.model is None:
            self.load_model()

        start_time = time.time()
        
        audio_data, sampling_rate = librosa.load(audio_path, sr=16000)
        inputs = self.feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode using tokenizer if available
        if self.tokenizer:
            text = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        else:
            # Fallback: simple join of predicted IDs
            text = " ".join([str(id.item()) for id in predicted_ids[0]])
        
        elapsed = time.time() - start_time

        return {
            "text": text,
            "processing_time": elapsed,
            "confidence": None
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model metadata."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "framework": "HuBERT"
        }
