import logging
import torch
import soundfile as sf
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

# Import Demucs
try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model as get_demucs_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# Import DeepFilterNet
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df.utils import download_model
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False

logger = logging.getLogger(__name__)

class AudioEnhancer(ABC):
    """Abstract base class for audio enhancement/cleaning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def enhance(self, input_path: Path, output_path: Path) -> None:
        """Applies enhancement to the audio file."""
        pass

class DemucsEnhancer(AudioEnhancer):
    """Wrapper for Demucs source separation (keeping vocals)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not DEMUCS_AVAILABLE:
            raise ImportError("Demucs is not installed.")
        
        model_name = self.config.get("model_name", "htdemucs")
        logger.info(f"Loading Demucs model: {model_name}")
        self.model = get_demucs_model(model_name)
        self.model.to(self.device)
        self.shifts = self.config.get("shifts", 1)
        self.overlap = self.config.get("overlap", 0.25)

    def enhance(self, input_path: Path, output_path: Path) -> None:
        # Load audio using soundfile to avoid torchcodec issues
        wav_np, sr = sf.read(str(input_path))
        wav = torch.from_numpy(wav_np).float()
        
        # Ensure shape is (channels, time)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.t()

        # Resample if necessary
        # if sr != self.model.samplerate:
        #     import julius
        #     wav = julius.resample_frac(wav, sr, self.model.samplerate)

        # Handle mono audio for stereo models (like htdemucs)
        if hasattr(self.model, 'audio_channels') and self.model.audio_channels == 2 and wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Apply model
        # ref = wav.mean(0)
        # wav = (wav - ref.mean()) / ref.std()
        
        # Add batch dimension
        wav = wav.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            sources = apply_model(
                self.model, 
                wav, 
                shifts=self.shifts, 
                split=True, 
                overlap=self.overlap, 
                progress=False,
                device=self.device
            )
        
        # sources shape: (batch, sources, channels, time)
        # We want the "vocals" source.
        # The order depends on the model, but usually ["drums", "bass", "other", "vocals"] for htdemucs
        source_names = self.model.sources
        try:
            vocals_idx = source_names.index("vocals")
        except ValueError:
            logger.warning("Vocals source not found in Demucs model. Using original audio.")
            return

        vocals = sources[0, vocals_idx].cpu()
        
        # Resample back to original SR if needed, or keep at model SR?
        # Usually ASR engines expect 16k. If we save at 44.1k, the ASR engine might handle it or not.
        # But our AudioProcessor standardizes to 16k.
        # Let's resample back to 16k (or original sr) to be consistent with the pipeline.
        # if self.model.samplerate != sr:
        #     import julius
        #     vocals = julius.resample_frac(vocals, self.model.samplerate, sr)

        # Save using soundfile
        # vocals shape is (channels, time), sf.write expects (time, channels)
        sf.write(str(output_path), vocals.t().numpy(), sr)

class DeepFilterNetEnhancer(AudioEnhancer):
    """Wrapper for DeepFilterNet noise suppression."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not DEEPFILTER_AVAILABLE:
            raise ImportError("DeepFilterNet is not installed.")
        
        # Load default model
        self.model, self.df_state, _ = init_df()

    def enhance(self, input_path: Path, output_path: Path) -> None:
        audio, _ = load_audio(str(input_path), sr=self.df_state.sr())
        
        # Denoise
        enhanced = enhance(self.model, self.df_state, audio)
        
        # Save
        save_audio(str(output_path), enhanced, self.df_state.sr())

