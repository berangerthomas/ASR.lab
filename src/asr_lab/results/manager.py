import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..analysis.report_interactive import InteractiveReportGenerator
from ..config.benchmark_config import ConfigLoader
from ..results.export import CsvExporter

logger = logging.getLogger(__name__)

class ResultManager:
    """
    Manages the storage, export, and reporting of benchmark results.
    
    Results are stored in JSON with pre-computed metrics for all
    normalization variants. The interactive HTML report allows
    switching between normalization options via checkboxes.
    """

    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.output_dir = Path("results")
        
        # Use config file name as subdirectory for reports
        self.config_name = config_loader.config_path.stem  # e.g., "quick_test" from "quick_test.yaml"
        self.reports_dir = self.output_dir / "reports" / self.config_name
        self.reports_dir.mkdir(exist_ok=True, parents=True)

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Saves the raw results to JSON.
        
        Results include pre-computed metrics_variants for all 8 normalization
        combinations, allowing the HTML report to switch between them.
        """
        logger.info(f"Saving {len(results)} results...")
        
        # Save raw results to a JSON file
        raw_results_path = self.reports_dir / "raw_results.json"
        try:
            with open(raw_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Raw results saved to {raw_results_path}")
        except Exception as e:
            logger.error(f"Failed to save raw results: {e}")

        # Export results to CSV
        try:
            csv_exporter = CsvExporter(results, self.reports_dir)
            csv_exporter.export()
            logger.info("Results exported to CSV.")
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

    def generate_reports(self, results: List[Dict[str, Any]]) -> None:
        """
        Generates interactive and static reports.
        
        Raises:
            RuntimeError: If report generation fails.
        """
        logger.info("Generating reports...")
        try:
            interactive_report = InteractiveReportGenerator(results, self.reports_dir)
            interactive_report.generate_report()
            logger.info("Interactive report generated.")
        except Exception as e:
            logger.error(f"Failed to generate interactive report: {e}")
            raise RuntimeError(f"Failed to generate interactive report: {e}") from e