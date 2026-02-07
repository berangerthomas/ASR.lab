# Changelog

## [1.0.0] - 2026-02-06

### Added

#### Text Normalization as Grid Search Dimension
- **Systematic text normalization**: Each transcription generates 2 results (raw + normalized)
- Normalized preset: lowercase + remove punctuation + normalize spaces
- Diff view shows both raw and normalized texts, if selected

#### Text normalizations
- Configurable text transforms: `ToLowerCase()`, `RemovePunctuation()`, `ExpandCommonEnglishContractions()`
- Metrics and transforms computed consistently with jiwer library

### Visual Encoding
- **Symbol** = Degradation type (circle = original, diamond = reverb, etc.)
- **Color** = Engine (whisper = blue, nemo = purple, etc.)
- **Size** = Text normalization (normalized = large, raw = small)

### Changed
- **wav2vec2 engine**: Output normalized to lowercase (was outputting uppercase)

---

## [0.8.0] - 2026-02-04

### Added

#### Core Architecture
- Modular ASR Engine System (`src/asr_lab/engines/`)
  - Abstract base class `ASREngine` for unified engine interface
  - Whisper engine (OpenAI)
  - Vosk engine (offline recognition)
  - Wav2Vec2 engine (Meta)
  - Seamless M4T engine (Meta)
  - Extensible API system (`engines/api/`)

#### Metrics System (`src/asr_lab/metrics/`)
- Word Error Rate (WER)
- Character Error Rate (CER)
- Match Error Rate (MER)
- Word Information Lost (WIL)
- Word Information Preserved (WIP)

#### Audio Processing (`src/asr_lab/audio/`)
- Audio degradation via VST plugins
- Audio enhancement: Demucs, DeepFilterNet
- Audio loader, normalization (EBU R128), processor, segmentation

#### Benchmarking Framework (`src/asr_lab/benchmarks/`)
- BenchmarkRunner, DataManager, Dataset, Experiment

#### Configuration System (`src/asr_lab/config/`)
- YAML-based configuration loader
- Dynamic engine and metric registries

#### Results Management (`src/asr_lab/results/`)
- Result storage, aggregation, export (JSON, CSV)

#### Analysis (`src/asr_lab/analysis/`)
- Interactive HTML reports (Plotly)
- Jinja2 templates

#### CLI (`src/asr_lab/cli/`)
- Click-based command-line interface

#### Core Models (`src/asr_lab/core/`)
- Pydantic models: EngineConfig, TranscriptionResult

### Experimental
- NeMo engine: does not work on Windows (SIGKILL)
- HuBERT engine: uses Wav2Vec2 tokenizer fallback

### Not Implemented
- Kaldi engine (requires external installation)
- USM engine (API not available)
- BLEU, Perplexity, RTF, Latency metrics
- Matplotlib visualizations
- Statistical analysis

### Known Issues
- NeMo: `signal.SIGKILL` not available on Windows
- HuBERT: uses Wav2Vec2 tokenizer fallback
- ~~HuBERT: outputs uppercase text~~ (fixed in 1.0.0: lowercase applied)
- ~~No text normalization before metric computation~~ (fixed in 1.0.0)

### Features
- Multi-language support (en, fr, de, es)
- YAML configuration
- GPU acceleration (CUDA)
- Batch processing
- Plugin architecture

### Technical
- Python 3.12+
- PyTorch backend
- Type hints

[0.8.0]: https://github.com/berangerthomas/ASR.lab/releases/tag/v0.8.0
