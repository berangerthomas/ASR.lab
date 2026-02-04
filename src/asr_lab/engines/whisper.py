import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult
from ..audio.loader import load_audio

class WhisperEngine(ASREngine):
    """
    An ASR engine implementation for the OpenAI Whisper model family.
    """

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_id = getattr(self.config, "model_id", None) or self.config.model_name
        if not self.model_id:
            raise ValueError("WhisperEngine config must include a 'model_id' or 'model_name'")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = None
        self.processor = None

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
        """Transcribes an audio file using the loaded Whisper model directly (bypassing pipeline)."""
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        # Load audio using centralized loader
        audio_data, sr = load_audio(audio_path, target_sr=16000)
        
        # Prepare generation arguments
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"
        
        chunk_length_s = getattr(self.config, "chunk_length_s", None)
        
        full_text = []
        
        if chunk_length_s:
            # Smart chunking (split on silence)
            for chunk_audio in self._smart_split(audio_data, sr, chunk_length_s):
                if len(chunk_audio) < sr * 0.1: # Skip tiny chunks
                    continue
                    
                try:
                    # Process chunk directly
                    inputs = self.processor(chunk_audio, sampling_rate=sr, return_tensors="pt").to(self.device)
                    
                    # Generate
                    with torch.no_grad():
                        predicted_ids = self.model.generate(inputs.input_features, **generate_kwargs)
                    
                    # Decode
                    transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    
                    if transcription:
                        full_text.append(transcription.strip())
                except Exception as e:
                    print(f"Error processing chunk: {e}")
        else:
            # No chunking (might truncate if > 30s)
            try:
                inputs = self.processor(audio_data, sampling_rate=sr, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    predicted_ids = self.model.generate(inputs.input_features, **generate_kwargs)
                
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                if transcription:
                    full_text.append(transcription.strip())
            except Exception as e:
                print(f"Error processing audio: {e}")

        elapsed = time.time() - start_time
        final_text = " ".join(full_text)

        return TranscriptionResult(
            text=final_text,
            processing_time=elapsed,
            confidence=None,
            metadata={"model": self.model_id}
        )

    def _smart_split(self, audio_data, sr, chunk_length_s=30, min_chunk_length_s=25):
        """
        Splits audio into chunks, trying to cut at the quietest point 
        between min_chunk_length_s and chunk_length_s.
        """
        total_samples = len(audio_data)
        chunk_samples = int(chunk_length_s * sr)
        min_chunk_samples = int(min_chunk_length_s * sr)
        
        cursor = 0
        while cursor < total_samples:
            # If remaining audio fits in one chunk, take it all
            if total_samples - cursor <= chunk_samples:
                yield audio_data[cursor:]
                break
            
            # Define the search window for splitting (e.g. between 25s and 30s)
            search_start = cursor + min_chunk_samples
            search_end = cursor + chunk_samples
            
            # Extract the search region
            search_region = audio_data[search_start:search_end]
            
            if len(search_region) == 0:
                # Should not happen due to check above, but safety first
                yield audio_data[cursor:cursor+chunk_samples]
                cursor += chunk_samples
                continue

            # Compute RMS energy to find silence
            # Frame length 512 (~32ms), hop length 128 (~8ms)
            try:
                rms = librosa.feature.rms(y=search_region, frame_length=512, hop_length=128)[0]
                
                # Find the frame with minimum energy
                min_idx = np.argmin(rms)
                
                # Convert frame index back to sample index
                # split_offset is relative to search_start
                split_offset = min_idx * 128 
                
                split_point = search_start + split_offset
            except Exception:
                # Fallback if librosa fails
                split_point = cursor + chunk_samples
            
            yield audio_data[cursor:split_point]
            cursor = split_point

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the Whisper model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype)
        }
