# Changelog

## [2.0.1] - 2026-03-21

### Removed

- **Code cleanup**: Removed unused API-based cloud engines (`azure.py`, `deepgram.py`, `google.py`) and inactive engine (`usm.py`).
- **Unused metrics**: Cleaned up orphans/irrelevant metrics modules (`bleu.py`, `perplexity.py`). Retained `rtf.py` and `latency.py` for potential future scaling.
- **Legacy reporting**: Removed old non-interactive static HTML reporting scripts and templates (`report.py`, `report_template.html`), as well as trailing backup files, fully transitioning to the modern interactive `report_interactive.py`.

## [2.0.0] - 2026-03-17

### Added

- **Moonshine engine** (`src/asr_lab/engines/moonshine.py`): new `MoonshineEngine` supporting `UsefulSensors/moonshine-base` and `UsefulSensors/moonshine-tiny`. Uses the Hugging Face Transformers pipeline. English-only; no additional dependencies required.
- **SenseVoice engine** (`src/asr_lab/engines/sensevoice.py`): new `SenseVoiceEngine` supporting `FunAudioLLM/SenseVoiceSmall` via the FunASR library. Multilingual with automatic language detection (`auto`, `zh`, `en`, `ja`, `ko`, `yue`). Applies Inverse Text Normalization by default.
- **Cross-language analysis**: new "Cross-Language Analysis" tab in the interactive report with grouped bar chart (engine × language), engine × language heatmap, language consistency chart, and aggregated statistics table.
- Language filter in the global filter bar.
- Language column in the results summary table and language badge in transcription cards.
- Language as a "Group By" / "Color By" option in box plots.
- Language included in heatmap row labels.
- `aggregator.py`: statistics module (`aggregate_by`, `cross_language_matrix`, `language_consistency`) for multi-file and cross-language aggregation.
- HTML report, tab "Detailled transcription analysis" : sort buttons and time metric.

### Changed

- **Client-Side Rendering**: Replaced server-side Plotly/Pandas plotting with JSON chart data (`chart_data_json`) for responsive client-side UI rendering.
- **Reporting System**: `InteractiveReportGenerator` now prepares a JSON-serializable records list instead of generating an HTML Plotly div.
- **Engine Metrics**: Updated `engine_registry`, `nemo`, `vosk`, `aggregator`, and `export` to expose explicit metadata/metrics required by the new client-side visualizations.
- CSV export now includes `enhancement`, `audio_norm`, and `text_norm` columns and reads `language` directly from results instead of parsing it from the dataset name.
- Scatter chart customdata includes language for filter support.
- **`audio_source_dir` now accepts a directory, a single file, or a glob pattern**:  
  - Directory (e.g. `"data/audio"`): loads all `*.json` manifest files in the directory.  
  - Single file (e.g. `"data/audio/manifest_en.json"`): loads that manifest only.  
  - Glob pattern (e.g. `"data/audio/manifest_fr*.json"`): loads all matching `.json` files.  
  Previously, only a directory was supported and a single hardcoded `manifest.json` was expected.
- Relative audio paths in manifests are now resolved from the **manifest's parent directory** (instead of from `audio_source_dir`).

### Fixed

- Fix box plot controls: "Normalization" option was not wired to any data attribute — replaced by distinct "Audio Norm" and "Text Norm" options matching the existing JS switch cases.
- Convert Demucs-separated vocals to mono before writing (average channels) and simplify saving logic to write a 1-D waveform. This ensures ASR pipelines receive mono audio and avoids incorrect transposes.
- Also import os and close the file descriptor returned by tempfile.mkstemp immediately to avoid descriptor leaks and allow the downloader to open the temp file by path.

### Removed

- **Server-side diffs**: Removed server-side character-level alignment and heavy Pandas usage in favor of a lazy JavaScript char-diff implementation.
- Delete unused `visualizer.py` (Matplotlib/Seaborn) and `visualizer_plotly.py` — all visualizations are now handled by the interactive report template via client-side JS.

---

## [1.1.0] - 2026-03-07

### Added

- Introduce a pre-flight engine setup system and related fixes/enhancements:
  - Add `src/asr_lab/setup`: `engine_setup`, `nemo_patch`, `vosk_setup` and package `__init__` to prepare engine-specific prerequisites at runtime (NeMo Windows SIGKILL compatibility patch and automatic Vosk model download+extraction).
  - `BenchmarkRunner` calls `ensure_engines_ready(...)` so engines are prepared before initialization.
- Update interactive report template: make filters bar sticky with backdrop blur and add `IntersectionObserver` to toggle shadow on scroll.

### Changed

- Improve `SeamlessM4T` engine: detect v2 models, select appropriate model class, and cap generation tokens (`max_new_tokens=256`); minor import adjustments.
- Move `deepfilternet` into an optional dependency group (`deepfilter`) in `pyproject.toml`.

---

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
