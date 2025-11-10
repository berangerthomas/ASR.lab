import soundfile as sf
import numpy as np
import pyloudnorm as pyln
from pathlib import Path
from typing import Dict, Any, Tuple

class AudioProcessor:
    """Handles audio processing tasks like loading, resampling, and normalization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def process_audio(self, input_path: Path, output_path: Path) -> None:
        """
        Loads an audio file, processes it according to the configuration,
        and saves it to the output path.
        """
        data, sr = self._load_and_prepare(input_path)
        
        if self.config.get('normalization', {}).get('enabled', False):
            data = self._normalize(data, sr)
            
        sf.write(output_path, data, self.config.get('sample_rate', 16000))

    def _load_and_prepare(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Loads audio, converts to mono, and resamples."""
        data, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample if necessary
        target_sr = self.config.get('sample_rate', 16000)
        if sr != target_sr:
            # Note: Using resample_poly from scipy would be more accurate
            # but this is a simpler approach for now.
            num_samples = int(len(data) * float(target_sr) / sr)
            data = np.interp(
                np.linspace(0., 1., num_samples, endpoint=False),
                np.linspace(0., 1., len(data), endpoint=False),
                data
            )
        
        return data, target_sr

    def _normalize(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Normalizes audio loudness to a target level."""
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(data)
        target_loudness = self.config.get('normalization', {}).get('target_loudness', -23.0)
        normalized_data = pyln.normalize.loudness(data, loudness, target_loudness)
        return normalized_data
