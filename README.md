# ASR.lab

A comprehensive benchmarking platform for automatic speech recognition (ASR) systems. This tool provides controlled audio degradation, audio enhancement, loudness normalization, multiple evaluation metrics, and comparative analysis across languages and model architectures.

## Overview

ASR.lab enables systematic evaluation of speech recognition engines under various acoustic conditions. It supports multiple ASR frameworks, applies configurable audio degradations, tests audio enhancement algorithms, and generates detailed performance reports with interactive visualizations.

## Features

- **Multi-Engine Support**: Compare performance across different ASR frameworks (Whisper, Wav2Vec2, NeMo, Vosk, SeamlessM4T, etc.)
- **Audio Degradation**: Apply controlled acoustic degradations (reverb, noise, compression) via VST3 plugins
- **Audio Enhancement**: Test denoising/enhancement algorithms (Demucs, DeepFilterNet) on degraded audio
- **Loudness Normalization**: Grid search across different LUFS normalization levels (EBU R128 compliant)
- **Evaluation Metrics**: WER, CER, MER, WIL, WIP for comprehensive transcription analysis
- **Interactive Reports**: HTML reports with sortable tables, Plotly visualizations, and multi-filter dropdowns
- **Multilingual**: Support for multiple languages and language-specific models
- **Extensible**: Plugin architecture for adding new engines and metrics
- **Grid Search**: Automatic Cartesian product of all test parameters (degradation × enhancement × normalization)

## Processing Pipeline

The benchmark pipeline processes audio in the following order:

```
Original Audio → Degradation (VST3) → Enhancement (Demucs) → Normalization (LUFS) → ASR Engine → Metrics
```

Each stage is optional and configurable. Multiple options at each stage create a grid search.

## Supported ASR Engines

| Engine | Status | Notes |
|--------|--------|-------|
| Whisper (OpenAI) | Tested | Multilingual, multiple model sizes |
| Wav2Vec 2.0 (Meta) | Tested | Language-specific fine-tuning |
| SeamlessM4T (Meta) | Tested | Multilingual translation and transcription |
| Vosk | Tested | Offline recognition, requires local model |
| HuBERT (Meta) | Experimental | Uses Wav2Vec2 tokenizer fallback |
| NeMo (NVIDIA) | Linux only | Does not work on Windows (SIGKILL) |
| Kaldi | Not implemented | Requires external Kaldi installation |
| USM (Google) | Not implemented | API not available |

## Evaluation Metrics

| Metric | Name | Description |
|--------|------|-------------|
| **WER** | Word Error Rate | Standard ASR metric: (S+D+I)/N |
| **CER** | Character Error Rate | Better for CJK languages |
| **MER** | Match Error Rate | Bounded version of WER (0-1) |
| **WIL** | Word Information Lost | Proportion of information lost |
| **WIP** | Word Information Preserved | Complement of WIL |

## Installation

### Requirements

- Python 3.12 or higher
- CUDA-capable GPU (optional, for accelerated inference)
- 8GB+ RAM recommended

### Basic Installation

```bash
git clone https://github.com/berangerthomas/ASR.lab.git
cd ASR.lab
pip install -e .
```

### Optional Dependencies

For specific engines, install additional packages:

```bash
# Vosk (offline recognition)
pip install vosk

# SeamlessM4T (multilingual)
pip install torchaudio

# NeMo (NVIDIA models)
pip install nemo_toolkit[asr]
```

## Configuration

Benchmarks are defined in YAML configuration files located in `configs/`.

See `configs/reference_complete.yaml` for a complete configuration reference.

### Basic Configuration Structure

```yaml
benchmark:
  name: "my_benchmark"
  description: "Description of the benchmark"

data:
  audio_source_dir: "data/audio"
  processed_dir: "data/processed"
  reference_dir: "data/references"

audio_processing:
  sample_rate: 16000
  channels: 1

# Loudness normalization (grid search)
normalizations:
  - name: "broadcast"
    enabled: true
    method: "lufs"
    target_loudness: -23.0
  - name: "no_norm"
    enabled: true
    method: "none"

# Audio degradation via VST3 plugins
degradations:
  vst_plugin_path: "path/to/reverb.vst3"
  presets:
    - name: "cathedral"
      preset_name: "Cathedral"

# Audio enhancement/denoising
enhancements:
  - name: "demucs"
    enabled: true
    method: "demucs"
    model_name: "htdemucs"

engines:
  whisper:
    - id: "whisper-tiny"
      model_id: "openai/whisper-tiny"
      enabled: true
      chunk_length_s: 30
  
  wav2vec2:
    - id: "wav2vec2-fr"
      model_id: "facebook/wav2vec2-large-xlsr-53-french"
      enabled: true

metrics:
  - name: "wer"
    enabled: true
  - name: "cer"
    enabled: true
```

## Usage

### Running a Benchmark

```bash
python main.py run --config configs/default.yaml
```

### Multi-Engine Comparison

Use the provided multi-engine configuration:

```bash
python main.py run --config configs/multi_engine.yaml
```

### Output

Results are saved to `results/reports/`:

- `report_interactive.html`: Interactive HTML report with visualizations
- `results.csv`: Raw results in CSV format
- `raw_results.json`: Complete results with metadata

### Viewing Results

Open `results/reports/report_interactive.html` in a web browser. The report includes:

- Interactive Plotly charts with multi-filter dropdowns (engine, degradation, enhancement, normalization)
- Sortable summary table (click column headers)
- Side-by-side transcription comparison with diff highlighting
- Performance metrics (WER, CER, processing time)

## Data Organization

### Option 1: File-based (Simple)

```text
data/
├── audio/           # Source audio files (WAV format)
├── processed/       # Degraded audio files
└── references/      # Reference transcriptions (TXT format)
```

Audio files should be named consistently with reference files:
- Audio: `fr_0.wav` (prefix `fr` indicates language)
- Reference: `fr_0.txt`

### Option 2: Manifest-based (Robust)

Create a `manifest.json` in your audio directory:

```json
[
  {
    "audio_filepath": "data/audio/sample1.wav",
    "text": "This is a sample transcription.",
    "lang": "en"
  },
  {
    "audio_filepath": "data/audio/sample2.wav",
    "text": "Ceci est une transcription.",
    "lang": "fr"
  }
]
```

## Extending the Platform

### Adding a New Engine

1. Create a new engine class in `src/asr_lab/engines/`:

```python
from .base import ASREngine
from pathlib import Path
from typing import Dict, Any

class MyEngine(ASREngine):
    def load_model(self) -> None:
        # Load model
        pass
    
    def transcribe(self, audio_path: Path, language: str) -> Dict[str, Any]:
        # Transcribe audio
        return {
            "text": transcription,
            "processing_time": elapsed,
            "confidence": None
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        return {"framework": "MyEngine"}
```

2. Register in `src/asr_lab/config/engine_registry.py`:

```python
from ..engines.my_engine import MyEngine

ENGINE_REGISTRY = {
    "my_engine": MyEngine,
    # ... other engines
}
```

## Performance Considerations

### GPU Acceleration

Most models automatically use CUDA if available. Check GPU usage:

```python
import torch
print(torch.cuda.is_available())
```

## Troubleshooting

### ModuleNotFoundError

Install missing dependencies:

```bash
pip install librosa transformers torch tqdm ruff
```

### CUDA Out of Memory

Reduce batch size or use smaller models. For Whisper, use tiny or base variants.

### Audio Format Errors

Ensure audio files are:
- WAV format
- 16kHz sample rate (or configure accordingly)
- Mono channel

### Vosk Model Not Found

Download models from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and extract the ZIP archives to `models/` directory.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome. Please submit pull requests or open issues for bugs and feature requests.
