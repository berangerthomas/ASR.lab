from pedalboard import load_plugin
import soundfile as sf
from pathlib import Path
import numpy as np

class AudioDegradation:
    """Applies audio degradations using VST plugins."""

    def __init__(self, plugin_path: str):
        self.vst_plugin = load_plugin(plugin_path)

    def apply_degradation(self, input_path: Path, output_path: Path, preset_name: str) -> None:
        """
        Applies a specific degradation preset to an audio file.
        """
        # Set the preset on the VST plugin.
        # Note: Pylance may flag this attribute as unknown, but this is the
        # correct way to set presets based on the library's usage in the notebook.
        self.vst_plugin.program = preset_name  # type: ignore
        
        # Read the audio file using soundfile
        audio, samplerate = sf.read(input_path)
        
        # Pedalboard expects a shape of (num_channels, num_samples).
        # Soundfile provides (num_samples,) for mono. Reshape for compatibility.
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

        # Apply the effect
        effected_audio = self.vst_plugin(audio, samplerate)
        
        # Soundfile expects (num_samples,) for mono. Squeeze the output from pedalboard.
        if effected_audio.shape[0] == 1:
            output_audio = effected_audio.squeeze()
        else:
            # For stereo, transpose from (num_channels, num_samples) to (num_samples, num_channels)
            output_audio = effected_audio.T

        # Write the processed audio to the output file
        sf.write(output_path, output_audio, samplerate)
