import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)

def load_audio(file_path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file, converts it to mono, and resamples it to the target sample rate.
    
    Args:
        file_path: Path to the audio file.
        target_sr: Target sample rate (default: 16000).
        
    Returns:
        Tuple containing:
            - Audio data as a numpy array (float32).
            - Sample rate (int).
            
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the audio file is invalid or cannot be read.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    file_path_str = str(path_obj)
    
    try:
        # Try loading with soundfile first (faster)
        audio, sr = sf.read(file_path_str)
    except Exception:
        try:
            # Fallback to librosa (supports more formats via ffmpeg/audioread)
            audio, sr = librosa.load(file_path_str, sr=None, mono=False)
        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {e}")

    # Convert to mono if stereo
    if audio.ndim > 1:
        # Check shape to determine channel dimension
        # Usually samples > channels
        if audio.shape[0] < audio.shape[1]:
            # Likely (channels, samples) - librosa style
            audio = np.mean(audio, axis=0)
        else:
            # Likely (samples, channels) - soundfile style
            audio = np.mean(audio, axis=1)
            
    # Resample if necessary
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    return audio.astype(np.float32), sr
