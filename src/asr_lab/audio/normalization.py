import soundfile as sf
import numpy as np
import pyloudnorm as pyln
import librosa
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AudioNormalizer:
    """
    Handles audio normalization using pyloudnorm.
    Also ensures audio is ASR-ready (mono, 16kHz).
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def _ensure_mono_and_resample(self, data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """
        Converts audio to mono and resamples to target sample rate.
        Returns (data, sample_rate).
        """
        # Convert to mono if stereo
        if data.ndim > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        elif data.ndim > 1:
            data = data.squeeze()
        
        # Resample if needed
        if sr != self.target_sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate
        
        return data, sr

    def prepare_for_asr(self, input_path: Path, output_path: Path) -> None:
        """
        Prepares audio for ASR without normalization.
        Ensures mono, 16kHz, 16-bit PCM format.
        """
        try:
            data, sr = sf.read(str(input_path))
            
            # Ensure float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Ensure mono and correct sample rate
            data, sr = self._ensure_mono_and_resample(data, sr)
            
            sf.write(str(output_path), data, sr, subtype='PCM_16')
            
        except Exception as e:
            logger.error(f"Failed to prepare {input_path} for ASR: {e}")
            raise e

    def normalize(self, input_path: Path, output_path: Path, target_loudness: float = -23.0) -> None:
        """
        Normalizes the audio at input_path to target_loudness LUFS and saves to output_path.
        Also ensures the output is mono and at the target sample rate.
        """
        try:
            data, sr = sf.read(str(input_path))
            
            # Ensure float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Handle multi-channel (pyloudnorm expects (samples, channels))
            # If mono, it expects (samples, 1) or (samples,) depending on version, usually (samples, channels) is safer
            # But our pipeline usually converts to mono earlier.
            # pyloudnorm works with shape (samples, channels)
            
            # Check if mono (1D array)
            if data.ndim == 1:
                # pyloudnorm expects 2D array (samples, channels)
                # We reshape (N,) to (N, 1)
                data = data.reshape(-1, 1)

            meter = pyln.Meter(sr) # create BS.1770 meter
            loudness = meter.integrated_loudness(data)
            
            # Normalize
            normalized_audio = pyln.normalize.loudness(data, loudness, target_loudness)
            
            # Check for clipping
            peak = np.max(np.abs(normalized_audio))
            if peak > 1.0:
                logger.warning(f"Normalization to {target_loudness} LUFS caused clipping (peak: {peak:.2f}). Limiting to -1.0 dBTP.")
                normalized_audio = normalized_audio / peak * 0.89125 # -1.0 dB
            
            # Convert to mono and resample for ASR compatibility
            normalized_audio, sr = self._ensure_mono_and_resample(normalized_audio, sr)
            
            sf.write(str(output_path), normalized_audio, sr, subtype='PCM_16')
            
        except Exception as e:
            logger.error(f"Failed to normalize {input_path}: {e}")
            # Fallback: copy original if normalization fails?
            # Or raise error?
            raise e
