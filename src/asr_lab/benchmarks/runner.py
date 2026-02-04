import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from ..config.benchmark_config import ConfigLoader
from ..config.engine_registry import ENGINE_REGISTRY
from ..config.metric_registry import METRIC_REGISTRY
from ..engines.base import ASREngine
from ..metrics.base import Metric
from ..results.manager import ResultManager
from .data_manager import DataManager

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Orchestrates the ASR benchmarking process using a modular architecture.
    """

    def __init__(self, config_path: Path):
        self.config_loader = ConfigLoader(config_path)
        self.data_manager = DataManager(self.config_loader)
        self.result_manager = ResultManager(self.config_loader)
        self.engines = self._initialize_engines()
        self.metrics = self._initialize_metrics()
        self.failed_engines: List[str] = []

    def run(self) -> None:
        """
        Executes the full benchmark pipeline.
        
        Raises:
            RuntimeError: If critical errors occur during the benchmark.
        """
        logger.info("Starting benchmark...")

        # 1. Prepare audio datasets
        datasets = self.data_manager.prepare_datasets()
        if not datasets:
            raise RuntimeError("No datasets prepared. Aborting benchmark.")

        results = []
        self.failed_engines = []

        # 2. Run experiments
        # We iterate over engines first to minimize model loading/unloading overhead
        for engine in self.engines:
            logger.info(f"Starting evaluation for engine: {engine.name}")

            try:
                engine.load_model()
            except Exception as e:
                logger.error(f"Failed to load model for engine {engine.name}: {e}")
                self.failed_engines.append(engine.name)
                continue

            # Process datasets for this engine
            engine_results = self._process_datasets_sequential(engine, datasets)

            results.extend(engine_results)

            # Optional: Explicitly unload model if method exists to free memory
            if hasattr(engine, "unload_model"):
                engine.unload_model()

        # 3. Save and visualize results
        if results:
            self.result_manager.save_results(results)
            self.result_manager.generate_reports(results)
        else:
            raise RuntimeError("No results to save. All engines may have failed.")

        # Log summary
        if self.failed_engines:
            logger.warning(f"Engines that failed to load: {', '.join(self.failed_engines)}")

        logger.info("Benchmark finished.")

    def _process_datasets_sequential(self, engine: ASREngine, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process datasets sequentially."""
        results = []
        for dataset in tqdm(datasets, desc=f"Evaluating {engine.name}"):
            try:
                result = self._process_single_item(engine, dataset)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {dataset['name']} with {engine.name}: {e}")
        return results

    def _process_single_item(self, engine: ASREngine, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single dataset item with a specific engine.
        This method is thread-safe if the engine's transcribe method is thread-safe.
        """
        # Skip if language not supported by engine (if engine has language constraints)
        # This logic could be added here if engines expose supported languages

        transcription_result = engine.transcribe(dataset["audio_path"], dataset["language"])

        metrics_results = {}
        for metric in self.metrics:
            try:
                metrics_results[metric.name] = metric.compute(
                    prediction=transcription_result.text, reference=dataset["reference_text"]
                )
            except Exception as e:
                logger.warning(f"Metric {metric.name} failed: {e}")
                metrics_results[metric.name] = None

        return {
            "dataset": dataset["name"],
            "engine": engine.name,
            "language": dataset.get("language", "unknown"),
            "degradation": dataset.get("degradation", "None"),
            "enhancement": dataset.get("enhancement", "None"),
            "normalization": dataset.get("normalization", "None"),
            "transcription": transcription_result.model_dump(),
            "metrics": metrics_results,
            "reference_text": dataset["reference_text"],
        }

    def _initialize_engines(self) -> List[ASREngine]:
        """Initializes the ASR engines specified in the config."""
        engines = []
        # With Pydantic, get_config() returns the BenchmarkConfig object
        config_obj = self.config_loader.get_config()
        engine_configs = config_obj.engines
        
        if not engine_configs:
            return []

        for engine_name, configs in engine_configs.items():
            if engine_name in ENGINE_REGISTRY:
                for config in configs:
                    if config.enabled:
                        try:
                            engine_class = ENGINE_REGISTRY[engine_name]
                            engines.append(engine_class(config))
                        except Exception as e:
                            logger.error(f"Failed to initialize engine {engine_name}: {e}")
            else:
                logger.warning(f"Engine '{engine_name}' not found in registry.")
        return engines

    def _initialize_metrics(self) -> List[Metric]:
        """Initializes the metrics specified in the config."""
        metrics = []
        # With Pydantic, get_config() returns the BenchmarkConfig object
        config_obj = self.config_loader.get_config()
        metric_configs = config_obj.metrics

        for config in metric_configs:
            metric_name = config.name
            if metric_name in METRIC_REGISTRY and config.enabled:
                try:
                    metric_class = METRIC_REGISTRY[metric_name]
                    # Pass the Pydantic model or convert to dict if metric expects dict
                    # Assuming metrics still expect dict for now, or we update them later.
                    # For minimal friction, let's pass dict.
                    metrics.append(metric_class(config.model_dump()))
                except Exception as e:
                    logger.error(f"Failed to initialize metric {metric_name}: {e}")
            else:
                if metric_name not in METRIC_REGISTRY:
                    logger.warning(f"Metric '{metric_name}' not found in registry.")
        return metrics