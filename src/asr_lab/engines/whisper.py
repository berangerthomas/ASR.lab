import time
import logging
import torch
import librosa
from pathlib import Path
from typing import Dict, Any
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperEngine(ASREngine):
    """
    An ASR engine implementation for the OpenAI Whisper model family.
    
    Uses the native Whisper long-form transcription with sequential decoding
    for accurate handling of audio files of any length.
    """

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_id = getattr(self.config, "model_id", None) or self.config.model_name
        if not self.model_id:
            raise ValueError("WhisperEngine config must include a 'model_id' or 'model_name'")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Loads the Whisper model and processor."""
        if self.model_id is None:
            raise ValueError("Model ID cannot be None.")
            
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_id}")
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            logger.info(f"Whisper model loaded on {self.device}")
            
    def unload_model(self) -> None:
        """Unloads the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        """
        Transcribes an audio file using Whisper's native long-form transcription.
        
        Uses sequential decoding with condition_on_prev_tokens for accurate
        transcription of audio files of any length.
        """
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Load audio with librosa (handles all formats, resamples to 16kHz)
            audio, sr = librosa.load(str(audio_path), sr=16000)
            
            # Process audio WITHOUT truncation for long-form support
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True
            ).to(self.device)
            
            # Prepare generation arguments
            generate_kwargs = {
                "return_timestamps": True,  # Required for long-form
                "condition_on_prev_tokens": True,  # Sequential decoding
            }
            
            if language:
                generate_kwargs["language"] = language
                generate_kwargs["task"] = "transcribe"
            
            # Add attention mask if available
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                generate_kwargs["attention_mask"] = inputs.attention_mask
            
            # Generate transcription
            generated_ids = self.model.generate(
                inputs.input_features,
                **generate_kwargs
            )
            
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            text = ""
        
        elapsed = time.time() - start_time

        return TranscriptionResult(
            text=text,
            processing_time=elapsed,
            confidence=None,
            metadata={"model": self.model_id}
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the Whisper model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype)
        }
