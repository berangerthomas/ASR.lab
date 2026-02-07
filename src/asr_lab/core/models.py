from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict

class TranscriptionResult(BaseModel):
    """Standardized result from an ASR engine."""
    text: str
    processing_time: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BenchmarkMeta(BaseModel):
    name: str
    description: str

class DataConfig(BaseModel):
    audio_source_dir: str = "data/audio"
    processed_dir: str = "data/processed"

class NormalizationConfig(BaseModel):
    name: str
    enabled: bool = True
    method: str = "lufs" # "lufs", "peak", or "none"
    target_loudness: float = -23.0

class AudioProcessingConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1

class DegradationPreset(BaseModel):
    name: str
    preset_name: str

class DegradationsConfig(BaseModel):
    vst_plugin_path: Optional[str] = None
    presets: List[DegradationPreset] = Field(default_factory=list)

class EnhancementConfig(BaseModel):
    name: Optional[str] = None
    enabled: bool = False
    method: str = "demucs" # "demucs" or "deepfilternet"
    model_name: Optional[str] = None
    model_config = ConfigDict(extra='allow')

class EngineConfig(BaseModel):
    id: str
    enabled: bool = True
    model_name: Optional[str] = None
    # Allow extra fields for engine-specific settings
    model_config = ConfigDict(extra='allow')

class MetricConfig(BaseModel):
    name: str
    enabled: bool = True
    # Allow extra fields for metric-specific settings
    model_config = ConfigDict(extra='allow')

class BenchmarkConfig(BaseModel):
    """Root configuration for the benchmark."""
    benchmark: BenchmarkMeta
    data: DataConfig = Field(default_factory=DataConfig)
    audio_processing: AudioProcessingConfig = Field(default_factory=AudioProcessingConfig)
    degradations: Optional[DegradationsConfig] = None
    enhancements: List[EnhancementConfig] = Field(default_factory=list)
    normalizations: List[NormalizationConfig] = Field(default_factory=list)
    engines: Dict[str, List[EngineConfig]]
    metrics: List[MetricConfig]
