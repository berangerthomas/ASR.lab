import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from .loader import load_audio

class AudioProcessor:
    """Handles audio processing tasks like loading and resampling."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def process_audio(self, input_path: Path, output_path: Path) -> None:
        """
        Loads an audio file, processes it according to the configuration,
        and saves it to the output path.
        """
        target_sr = self.config.get('sample_rate', 16000)
        data, sr = load_audio(input_path, target_sr=target_sr)
        sf.write(output_path, data, sr)

    def _load_and_prepare(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Loads audio, converts to mono, and resamples."""
        target_sr = self.config.get('sample_rate', 16000)
        return load_audio(audio_path, target_sr=target_sr)

