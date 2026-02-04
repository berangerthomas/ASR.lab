import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    start: float
    end: float
    speaker: str = "SPEAKER_00"

class VADSegmenter:
    """
    Uses Silero VAD to segment audio into speech chunks.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.utils = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info("Loading Silero VAD model...")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
            self.model = model.to(self.device)
            self.utils = utils
            logger.info("Silero VAD loaded.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def get_segments(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Returns a list of speech segments.
        audio: numpy array (float32), mono.
        sr: sample rate (must be 16000 for Silero usually, but it supports 8k).
        """
        (get_speech_timestamps, _, _, _, _) = self.utils
        
        # Convert to tensor
        wav = torch.from_numpy(audio).float()
        if wav.ndim > 1:
            wav = wav.mean(dim=0) # Ensure mono
        wav = wav.to(self.device)

        # Get timestamps
        # Silero expects normalized audio? It usually handles it.
        # But let's ensure it's 16k if possible.
        if sr != 16000:
            logger.warning(f"VAD input sr is {sr}, Silero prefers 16000. Results might be suboptimal.")
        
        speech_timestamps = get_speech_timestamps(
            wav, 
            self.model, 
            sampling_rate=sr,
            return_seconds=True
        )
        
        segments = []
        for ts in speech_timestamps:
            segments.append({
                "start": ts['start'],
                "end": ts['end'],
                "speaker": "SPEAKER_00"
            })
            
        return segments

class ChunkProcessor:
    """
    Logic to merge small segments into larger chunks for transcription.
    Ported from Stellascript.
    """
    TARGET_CHUNK_DURATION_S = 25.0
    MAX_CHUNK_DURATION_S = 29.0
    MIN_SILENCE_GAP_S = 0.5

    @staticmethod
    def chunk_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []

        logger.info(f"Chunking {len(segments)} segments...")
        final_chunks = []
        current_chunk = []
        current_chunk_duration = 0.0

        for i, seg in enumerate(segments):
            seg_duration = seg['end'] - seg['start']
            
            # Check if adding this segment exceeds max duration
            if current_chunk and (current_chunk_duration + seg_duration > ChunkProcessor.MAX_CHUNK_DURATION_S):
                # Finalize current chunk
                final_chunks.append(ChunkProcessor._merge_group(current_chunk))
                current_chunk = []
                current_chunk_duration = 0.0
            
            current_chunk.append(seg)
            current_chunk_duration += seg_duration
            
            # Check if we reached target duration
            if current_chunk_duration >= ChunkProcessor.TARGET_CHUNK_DURATION_S:
                # Look for a split point (long silence)
                is_last = (i == len(segments) - 1)
                if not is_last:
                    next_seg = segments[i+1]
                    silence = next_seg['start'] - seg['end']
                    if silence > ChunkProcessor.MIN_SILENCE_GAP_S:
                        # Good split point
                        final_chunks.append(ChunkProcessor._merge_group(current_chunk))
                        current_chunk = []
                        current_chunk_duration = 0.0
        
        if current_chunk:
            final_chunks.append(ChunkProcessor._merge_group(current_chunk))
            
        return final_chunks

    @staticmethod
    def _merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {}
        
        start = group[0]['start']
        end = group[-1]['end']
        return {"start": start, "end": end, "segments": group}
