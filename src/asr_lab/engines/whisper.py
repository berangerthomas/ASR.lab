import time
import torch
import librosa
from pathlib import Path
from typing import Dict, Any
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines import pipeline

from .base import ASREngine

class WhisperEngine(ASREngine):
    """
    An ASR engine implementation for the OpenAI Whisper model family.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_id = self.config.get("model_id")
        if not self.model_id:
            raise ValueError("WhisperEngine config must include a 'model_id'")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = None
        self.processor = None
        self.pipe = None

    def load_model(self) -> None:
        """Loads the Whisper model and processor."""
        if self.model_id is None:
            raise ValueError("Model ID cannot be None.")
            
        if self.model is None:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )

    def transcribe(self, audio_path: Path, language: str) -> Dict[str, Any]:
        """Transcribes an audio file using the loaded Whisper model."""
        if self.pipe is None:
            self.load_model()
        
        # Ensure self.pipe is callable
        if not callable(self.pipe):
            raise TypeError("Transcription pipeline is not initialized correctly.")

        start_time = time.time()
        
        # Load the audio data first to avoid transformers' dependency on ffmpeg
        audio_data, sampling_rate = librosa.load(audio_path, sr=16000)
        
        output = self.pipe({"raw": audio_data, "sampling_rate": sampling_rate}, return_timestamps=True)
        
        elapsed = time.time() - start_time

        # Ensure output is a dictionary before accessing keys
        text = output.get("text", "") if isinstance(output, dict) else ""

        return {
            "text": text,
            "processing_time": elapsed,
            "confidence": None  # Whisper doesn't provide a single confidence score
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the Whisper model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype)
        }
