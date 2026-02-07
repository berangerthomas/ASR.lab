# ASR.lab

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/berangerthomas/ASR.lab-demo)

A comprehensive benchmarking platform for automatic speech recognition (ASR) systems. This tool provides controlled audio degradation, audio enhancement, loudness normalization, multiple evaluation metrics, and comparative analysis across languages and model architectures.

## Live Demo

Try the interactive visualization dashboard online:  
👉 **[ASR.lab Demo on Hugging Face Spaces](https://huggingface.co/spaces/berangerthomas/ASR.lab-demo)**

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
| Whisper (OpenAI) | ✅ Tested | Multilingual, long-form transcription support |
| Wav2Vec 2.0 (Meta) | ✅ Tested | Language-specific fine-tuning, outputs normalized to lowercase |
| SeamlessM4T (Meta) | ✅ Tested | Multilingual translation and transcription |
| NeMo (NVIDIA) | ✅ Tested | Windows support via SIGTERM patch (see below) |
| Vosk | ✅ Tested | Offline recognition, requires local model |
| HuBERT (Meta) | ⚠️ Experimental | Uses Wav2Vec2 tokenizer fallback |

## Evaluation Metrics

| Metric | Name | Description |
|--------|------|-------------|
| **WER** | Word Error Rate | Standard ASR metric: (S+D+I)/N |
| **CER** | Character Error Rate | Better for CJK languages |
| **MER** | Match Error Rate | Bounded version of WER (0-1) |
| **WIL** | Word Information Lost | Proportion of information lost |
| **WIP** | Word Information Preserved | Complement of WIL |

### Text Normalization (Grid Search Dimension)

Text normalization is applied **systematically** as a grid search dimension. Each transcription generates **two results**:

| Preset | Description |
|--------|-------------|
| **raw** | Texte brut, aucune transformation |
| **normalized** | Minuscules + sans ponctuation + espaces normalisés |

**Normalized preset applies:**
- **Lowercase conversion**: Case-insensitive comparison
- **Punctuation removal**: Ignores punctuation differences  
- **Whitespace normalization**: Collapses multiple spaces, trims

In the interactive report, use the **"Texte"** dropdown to:
- View **both** raw and normalized results side-by-side
- Filter to **raw only** to see exact ASR output
- Filter to **normalized only** for standard ASR evaluation

**Visual Encoding in Interactive Reports:**
- **Symbol** = Degradation type (circle = original, diamond = reverb, etc.)
- **Color** = Engine (whisper, nemo, etc.)
- **Size** = Text normalization (Large = Normalized, Small = Raw)

## Installation

### Requirements

- Python 3.12 or higher
- Optional CUDA-capable GPU

### Basic Installation

```bash
git clone https://github.com/berangerthomas/ASR.lab.git
cd ASR.lab
uv sync
```

## Configuration

Benchmarks are defined in YAML configuration files located in `configs/`.

See `configs/example.yaml` for a complete configuration reference.

### Basic Configuration Structure

```yaml
benchmark:
  name: "my_benchmark"
  description: "Description of the benchmark"

data:
  audio_source_dir: "data/audio"
  processed_dir: "data/processed"

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
python main.py run -c configs/default.yaml
```

This will:
1. Run transcriptions with all configured engines
2. Compute metrics for both raw and normalized text (grid search dimension)
3. Generate an interactive HTML report with dropdown filters including text normalization

### Interactive Report Features

The generated `report_interactive.html` includes:

- **Interactive Metric Normalization**: Toggle lowercase, punctuation removal, and contraction expansion - metrics update instantly without regenerating the report
- **Multi-filter dropdowns**: Filter by engine, degradation, enhancement, and normalization
- **Sortable tables**: Click column headers to sort
- **Side-by-side diff**: Compare reference and hypothesis with error highlighting
- **Multiple visualizations**: Scatter plots, heatmaps, box plots

### Output

Results are saved to `results/reports/<config>/`:

- `report_interactive.html`: Interactive HTML report with embedded metrics variants
- `results.csv`: Results in CSV format
- `raw_results.json`: JSON backup with reference and hypothesis text

### Viewing Results

Open `results/reports/<config>/report_interactive.html` in a web browser. The report includes:

- Interactive Plotly charts with multi-filter dropdowns (engine, degradation, enhancement, normalization)
- Sortable summary table (click column headers)
- Side-by-side transcription comparison with diff highlighting
- Performance metrics (WER, CER, processing time)

## Data Organization

### Manifest File (Required)

Create a `manifest.json` in your audio directory with all audio files and their reference transcriptions:

```json
[
  {
    "audio_filepath": "sample1.wav",
    "text": "This is a sample transcription.",
    "lang": "en"
  },
  {
    "audio_filepath": "sample2.wav",
    "text": "Ceci est une transcription.",
    "lang": "fr"
  }
]
```

**Required fields:**
- `audio_filepath`: Path to audio file (relative to manifest location or absolute)
- `text`: Reference transcription (ground truth)
- `lang`: Language code (ISO 639-1: en, fr, de, es, etc.)

**Audio format requirements:**
- WAV format recommended
- 16kHz sample rate (or configure in `audio_processing.sample_rate`)
- Mono channel recommended
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

### NeMo on Windows (SIGKILL Error)

NeMo uses `signal.SIGKILL` which doesn't exist on Windows. To fix:

1. Locate the file `.venv\Lib\site-packages\nemo\utils\exp_manager.py`
2. Find line ~170: `rank_termination_signal: signal.Signals = signal.SIGKILL`
3. Replace with: `rank_termination_signal: signal.Signals = signal.SIGTERM if not hasattr(signal, 'SIGKILL') else signal.SIGKILL`

Or run this one-liner:
```powershell
(Get-Content .venv\Lib\site-packages\nemo\utils\exp_manager.py) -replace 'signal\.SIGKILL', 'signal.SIGTERM if not hasattr(signal, "SIGKILL") else signal.SIGKILL' | Set-Content .venv\Lib\site-packages\nemo\utils\exp_manager.py
```

> **Note**: This patch is local to your venv and will be lost if you reinstall nemo_toolkit.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome. Please submit pull requests or open issues for bugs and feature requests.
