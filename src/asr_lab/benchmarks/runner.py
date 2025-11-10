from pathlib import Path
from typing import Dict, Any, List

from ..config.benchmark_config import ConfigLoader
from ..config.engine_registry import ENGINE_REGISTRY
from ..config.metric_registry import METRIC_REGISTRY
from ..engines.base import ASREngine
from ..metrics.base import Metric
from ..audio.processor import AudioProcessor
from ..audio.degradation import AudioDegradation
from ..analysis.report_interactive import InteractiveReportGenerator
from ..results.export import CsvExporter

class BenchmarkRunner:
    """
    Orchestrates the ASR benchmarking process.
    """

    def __init__(self, config_path: Path):
        self.config_loader = ConfigLoader(config_path)
        self.engines = self._initialize_engines()
        self.metrics = self._initialize_metrics()
        self.audio_processor = self._initialize_audio_processor()
        self.audio_degradation = self._initialize_audio_degradation()

    def run(self) -> None:
        """
        Executes the full benchmark pipeline.
        """
        print("Starting benchmark...")
        
        # 1. Prepare audio datasets
        datasets = self._prepare_datasets()

        # 2. Run experiments
        results = []
        for dataset in datasets:
            for engine in self.engines:
                print(f"Running experiment: [Dataset: {dataset['name']}, Engine: {engine.name}]")
                
                # Load the model once per engine
                engine.load_model()
                
                transcription_result = engine.transcribe(
                    dataset["audio_path"],
                    dataset["language"]
                )
                
                metrics_results = {}
                for metric in self.metrics:
                    metrics_results[metric.name] = metric.compute(
                        prediction=transcription_result["text"],
                        reference=dataset["reference_text"]
                    )
                
                results.append({
                    "dataset": dataset["name"],
                    "engine": engine.name,
                    "transcription": transcription_result,
                    "metrics": metrics_results,
                    "reference_text": dataset["reference_text"]
                })
        
        # 3. Save and visualize results
        output_dir = Path(self.config_loader.get_parameter("results", "output_dir", "results"))
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(exist_ok=True, parents=True)

        # Save raw results to a JSON file
        raw_results_path = reports_dir / "raw_results.json"
        import json
        with open(raw_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nRaw results saved to {raw_results_path}")

        # Generate interactive report (all-in-one)
        interactive_report = InteractiveReportGenerator(results, reports_dir)
        interactive_report.generate_report()

        # Export results to CSV
        csv_exporter = CsvExporter(results, reports_dir)
        csv_exporter.export()
        
        print("\nBenchmark finished.")

    def _prepare_datasets(self) -> List[Dict[str, Any]]:
        """
        Prepares the audio datasets for benchmarking, including degradations.
        Returns a list of dataset specifications.
        """
        print("Preparing datasets...")
        datasets = []
        
        # Get data paths from config
        audio_source_dir = Path(self.config_loader.get_parameter("data", "audio_source_dir", "data/audio"))
        reference_dir = Path(self.config_loader.get_parameter("data", "reference_dir", "data/references"))
        processed_dir = Path(self.config_loader.get_parameter("data", "processed_dir", "data/processed"))
        processed_dir.mkdir(exist_ok=True)

        # Find all audio files in the source directory
        for source_audio_path in audio_source_dir.glob("*.wav"):
            lang_code = source_audio_path.stem.split('_')[0]
            reference_text_path = reference_dir / f"{source_audio_path.stem}.txt"

            if not reference_text_path.is_file():
                print(f"Warning: Reference text not found for audio file: {source_audio_path}")
                continue

            try:
                with open(reference_text_path, 'r', encoding='utf-8') as f:
                    reference_text = f.read()
            except FileNotFoundError:
                print(f"Warning: Could not read reference text at: {reference_text_path}")
                continue

            # 1. Process the original (original) audio file
            processed_audio_path = processed_dir / f"{source_audio_path.stem}_original.wav"
            self.audio_processor.process_audio(source_audio_path, processed_audio_path)
            datasets.append({
                "name": f"{source_audio_path.stem}_original",
                "language": lang_code,
                "audio_path": processed_audio_path,
                "reference_text": reference_text
            })

            # 2. Apply degradations if configured
            if self.audio_degradation:
                degradations = self.config_loader.get_section("degradations").get("presets", [])
                for preset in degradations:
                    degraded_path = processed_dir / f"{source_audio_path.stem}_degraded_{preset['name']}.wav"
                    print(f"Applying degradation '{preset['name']}' to {source_audio_path.name}...")
                    self.audio_degradation.apply_degradation(
                        processed_audio_path,
                        degraded_path,
                        preset["preset_name"]
                    )
                    datasets.append({
                        "name": f"{source_audio_path.stem}_degraded_{preset['name']}",
                        "language": lang_code,
                        "audio_path": degraded_path,
                        "reference_text": reference_text
                    })
        
        print(f"Prepared {len(datasets)} datasets.")
        return datasets

    def _initialize_engines(self) -> List[ASREngine]:
        """Initializes the ASR engines specified in the config."""
        engines = []
        engine_configs = self.config_loader.get_section("engines")
        for engine_name, configs in engine_configs.items():
            if engine_name in ENGINE_REGISTRY:
                for config in configs:
                    if config.get("enabled", False):
                        engine_class = ENGINE_REGISTRY[engine_name]
                        engines.append(engine_class(config))
            else:
                print(f"Warning: Engine '{engine_name}' not found in registry.")
        return engines

    def _initialize_metrics(self) -> List[Metric]:
        """Initializes the metrics specified in the config."""
        metrics = []
        metric_configs = self.config_loader.get_section("metrics")
        
        # Ensure metric_configs is a list of dictionaries
        if not isinstance(metric_configs, list):
            print("Warning: 'metrics' section in config is not a list.")
            return []

        for config in metric_configs:
            if isinstance(config, dict):
                metric_name = config.get("name")
                if metric_name in METRIC_REGISTRY and config.get("enabled", False):
                    metric_class = METRIC_REGISTRY[metric_name]
                    # Pass the full config including 'name' so metric.name is set correctly
                    metrics.append(metric_class(config))
                else:
                    print(f"Warning: Metric '{metric_name}' not found or not enabled.")
            else:
                print(f"Warning: Invalid item in metrics configuration: {config}")
        return metrics

    def _initialize_audio_processor(self) -> AudioProcessor:
        """Initializes the audio processor."""
        audio_config = self.config_loader.get_section("audio_processing")
        return AudioProcessor(audio_config)

    def _initialize_audio_degradation(self) -> AudioDegradation | None:
        """Initializes the audio degradation module if configured."""
        degradation_config = self.config_loader.get_section("degradations")
        
        # The configuration might be a list of dictionaries.
        # For now, we'll assume a simple structure where the plugin path is at the top level.
        # This can be expanded to handle more complex degradation pipelines.
        if isinstance(degradation_config, dict):
            plugin_path = degradation_config.get("vst_plugin_path")
            if plugin_path:
                return AudioDegradation(plugin_path)
        return None
