# ASR.lab

A comprehensive benchmarking platform for automatic speech recognition (ASR) systems. This tool provides controlled audio degradation, multiple evaluation metrics, and comparative analysis across languages and model architectures.

## Overview

ASR.lab enables systematic evaluation of speech recognition engines under various acoustic conditions. It supports multiple ASR frameworks, applies configurable audio degradations, and generates detailed performance reports with interactive visualizations.

## Features

- **Multi-Engine Support**: Compare performance across different ASR frameworks
- **Audio Degradation**: Apply controlled acoustic degradations (reverb, noise, compression)
- **Evaluation Metrics**: Word Error Rate (WER), Character Error Rate (CER), and more
- **Interactive Reports**: HTML reports with sortable tables and Plotly visualizations
- **Multilingual**: Support for multiple languages and language-specific models
- **Extensible**: Plugin architecture for adding new engines and metrics

## Supported ASR Engines

### Open-Source Models

- **Whisper** (OpenAI): Multilingual transformer-based model with high accuracy
- **Wav2Vec 2.0** (Meta): Self-supervised learning model with strong performance
- **HuBERT** (Meta): Masked prediction model for speech representation
- **SeamlessM4T** (Meta): Multilingual multimodal translation and transcription
- **Vosk** (Alpha Cephei): Lightweight offline recognition based on Kaldi
- **NeMo** (NVIDIA): Suite including Parakeet, QuartzNet, and Conformer models. Now includes French ASR-trained models like `stt_fr_conformer_ctc_large`.
- **Kaldi** (External): Traditional GMM-HMM and DNN framework via subprocess

### Cloud API Support

- **Google Cloud Speech-to-Text**
- **Azure Speech Services**
- **Deepgram**

## Installation

### Requirements

- Python 3.12 or higher
- CUDA-capable GPU (optional, for accelerated inference)
- 8GB+ RAM recommended

### Basic Installation

```bash
git clone https://github.com/yourusername/ASR.lab.git
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
  normalization:
    enabled: true
    target_loudness: -22.0

degradations:
  vst_plugin_path: "path/to/reverb.vst3"
  presets:
    - name: "cathedral"
      preset_name: "Cathedral"

engines:
  whisper:
    - id: "whisper-tiny"
      model_id: "openai/whisper-tiny"
      enabled: true
  
  wav2vec2:
    - id: "wav2vec2-fr"
      model_id: "facebook/wav2vec2-large-xlsr-53-french"
      enabled: true

metrics:
  - name: "wer"
    enabled: true
```

### Engine-Specific Configuration

#### Whisper

```yaml
whisper:
  - id: "whisper-base"
    model_id: "openai/whisper-base"
    enabled: true
```

Available models: `whisper-tiny`, `whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large-v3`

#### Wav2Vec2

```yaml
wav2vec2:
  - id: "wav2vec2-fr"
    model_id: "facebook/wav2vec2-large-xlsr-53-french"
    enabled: true
```

Compatible with any Wav2Vec2 model from Hugging Face Hub.

#### HuBERT

```yaml
hubert:
  - id: "hubert-large"
    model_id: "facebook/hubert-large-ls960-ft"
    enabled: true
```

#### Vosk

Vosk requires downloading and extracting pre-trained models manually.

1. Download models from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
2. Extract ZIP archives to `models/` directory:

```bash
# Windows (PowerShell)
Expand-Archive -Path vosk-model-fr-0.22.zip -DestinationPath models/

# Linux/Mac
unzip vosk-model-fr-0.22.zip -d models/
```

3. Configure in YAML:

```yaml
vosk:
  - id: "vosk-fr"
    model_path: "models/vosk-model-fr-0.22"
    enabled: true
```

The extracted folder structure should contain:

```
models/
└── vosk-model-fr-0.22/
    ├── am/
    ├── conf/
    ├── graph/
    └── ivector/
```

Recommended models:

- French: `vosk-model-fr-0.22` (1.4GB)
- English: `vosk-model-en-us-0.22` (1.8GB)
- Small models: `vosk-model-small-fr-0.22` (41MB) for resource-constrained environments

#### NeMo

```yaml
nemo:
  - id: "nemo-fr-conformer-ctc-large"
    model_name: "stt_fr_conformer_ctc_large"
    enabled: true
```

Available French models:

- `stt_fr_conformer_ctc_large`: High-accuracy French ASR model
- `stt_fr_conformer_transducer_large`: Alternative transducer architecture
- `stt_fr_fastconformer_hybrid_large_pc`: FastConformer with hybrid CTC/Transducer

Available English models:

- `stt_en_conformer_ctc_large`: English Conformer CTC model
- `stt_en_fastconformer_ctc_large`: FastConformer for English

Requires `nemo_toolkit[asr]` installation.

#### Kaldi

Kaldi requires external installation and custom scripts.

```yaml
kaldi:
  - id: "kaldi-aspire"
    model_dir: "/path/to/kaldi/model"
    script_path: "/path/to/decode_script.sh"
    enabled: true
```

The script should accept audio file path and model directory as arguments and output transcription to stdout.

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

- Interactive Plotly charts with engine and degradation filters
- Sortable summary table (click column headers)
- Side-by-side transcription comparison
- Performance metrics (WER, processing time)

## Audio Degradation

The platform supports controlled audio degradation through VST3 plugins.

### VST Plugin Installation

You can install VST3 plugins in two ways:

1. **System-Wide Installation (Recommended for most users)**:
   - **Windows**: Place the `.vst3` file in `C:/Program Files/Common Files/VST3/`.
   - **macOS**: Place the `.vst3` file in `/Library/Audio/Plug-Ins/VST3/`.
   - **Linux**: Place the `.vst3` file in `~/.vst3/` or `/usr/lib/vst3/`.

2. **Local Project Installation (for portability)**:
   - Place the `.vst3` file directly into the `vst_plugins/` directory at the root of this project.
   - This method is ideal for ensuring that all project contributors use the exact same plugin version.

### Configuration in YAML

After installation, specify the path to the `.vst3` file in your configuration file (`configs/*.yaml`).

**Example for a system-wide plugin:**

```yaml
degradations:
  vst_plugin_path: "C:/Program Files/Common Files/VST3/PlaceIt.vst3"
  presets:
    - name: "cathedral"
      preset_name: "Cathedral"
```

**Example for a local project plugin:**

```yaml
degradations:
  vst_plugin_path: "vst_plugins/PlaceIt.vst3"
  presets:
    - name: "cathedral"
      preset_name: "Cathedral"
```

### Pristine Audio

Pristine (unmodified) audio is automatically included for baseline comparison.

## Metrics

### Word Error Rate (WER)

Primary metric for ASR evaluation. Calculates insertions, deletions, and substitutions at word level.

```yaml
metrics:
  - name: "wer"
    enabled: true
```

### Character Error Rate (CER)

Character-level accuracy metric.

```yaml
metrics:
  - name: "cer"
    enabled: true
```

### Additional Metrics

- **MER**: Match Error Rate
- **RTF**: Real-Time Factor (processing speed)
- **Latency**: Processing time in seconds

## Data Organization

```text
data/
├── audio/           # Source audio files (WAV format)
├── processed/       # Degraded audio files
└── references/      # Reference transcriptions (TXT format)
```

Audio files should be named consistently with reference files:

- Audio: `fr_0.wav`
- Reference: `fr_0.txt`

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

### Adding a New Metric

1. Create metric class in `src/asr_lab/metrics/`:

```python
from .base import Metric

class MyMetric(Metric):
    def compute(self, reference: str, hypothesis: str) -> float:
        # Compute metric
        return score
```

2. Register in `src/asr_lab/config/metric_registry.py`

## Performance Considerations

### GPU Acceleration

Most models automatically use CUDA if available. Check GPU usage:

```python
import torch
print(torch.cuda.is_available())
```

### Memory Requirements

Approximate memory requirements:

- Whisper tiny/base: 1-2GB
- Whisper small/medium: 2-5GB
- Whisper large: 10GB+
- Wav2Vec2/HuBERT: 3-5GB
- SeamlessM4T: 8GB+

### Processing Speed

Real-time factor (RTF) varies by model:

- Vosk: 0.1-0.3x (faster than real-time)
- Whisper tiny: 0.3-0.5x
- Whisper small/medium: 0.5-1.5x
- Large models: 2-5x (slower than real-time)

## Troubleshooting

### ModuleNotFoundError

Install missing dependencies:

```bash
pip install librosa transformers torch
```

### CUDA Out of Memory

Reduce batch size or use smaller models. For Whisper, use tiny or base variants.

### Audio Format Errors

Ensure audio files are:

- WAV format
- 16kHz sample rate (or configure accordingly)
- Mono channel

Convert using ffmpeg:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Vosk Model Not Found

Download models from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and extract the ZIP archives to `models/` directory. Do not leave models as ZIP files - they must be extracted folders containing `am/`, `conf/`, `graph/`, and `ivector/` subdirectories.

Example extraction:

```bash
# Windows
Expand-Archive vosk-model-fr-0.22.zip -DestinationPath models/

# Verify structure
ls models/vosk-model-fr-0.22/
# Should show: am, conf, graph, ivector
```

## License

See LICENSE file for details.

## References

- Whisper: [OpenAI Whisper Paper](https://cdn.openai.com/papers/whisper.pdf)
- Wav2Vec 2.0: [Facebook Research](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)
- HuBERT: [Paper on arXiv](https://arxiv.org/abs/2106.07447)
- Vosk: [Alpha Cephei](https://alphacephei.com/vosk/)

## Contributing

Contributions are welcome. Please submit pull requests or open issues for bugs and feature requests.
