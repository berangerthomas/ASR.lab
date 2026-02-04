import jinja2
from pathlib import Path
from typing import List, Dict, Any
import datetime

class ReportGenerator:
    """
    Generates an HTML report from benchmark results.
    """

    def __init__(self, results: List[Dict[str, Any]], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.template_dir = Path(__file__).parent / "templates"

    def generate_report(self):
        """
        Renders the HTML report from a Jinja2 template.
        """
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_dir))
        template = env.get_template("report_template.html")

        # Check which plots exist
        plot_files = {
            "interactive_dashboard": (self.output_dir / "interactive_dashboard.html").exists(),
            "wer_vs_time_combined": (self.output_dir / "wer_vs_time_combined.png").exists(),
            "wer_vs_time_original": (self.output_dir / "wer_vs_time_original.png").exists(),
            "wer_vs_time_degraded": (self.output_dir / "wer_vs_time_degraded.png").exists(),
        }

        # Prepare data for the template
        template_data = {
            "generation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": self._prepare_results_for_template(),
            "plots": plot_files,
        }

        # Render the template
        html_content = template.render(template_data)

        # Save the report
        report_path = self.output_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {report_path}")

    def _prepare_results_for_template(self) -> List[Dict[str, Any]]:
        """Prepares the results data for easy rendering in the template."""
        prepared = []
        for res in self.results:
            language = res.get("language") or res["dataset"].split('_')[0]
            degradation = res.get("degradation")
            enhancement = res.get("enhancement", "None")
            normalization = res.get("normalization")
            if not normalization:
                # Fallback: try to parse from dataset name
                dataset_name = res["dataset"]
                if dataset_name.endswith("norm_minus_18"):
                    normalization = "norm_minus_18"
                elif dataset_name.endswith("no_norm"):
                    normalization = "no_norm"
                else:
                    normalization = "None"
            
            if not degradation:
                is_pristine = "original" in res["dataset"]
                degradation = "original" if is_pristine else res["dataset"].split("degraded_")[-1]

            # Get WER value (try both lowercase and uppercase for backward compatibility)
            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")

            prepared.append({
                "dataset": res["dataset"],
                "engine": res["engine"],
                "language": language,
                "degradation": degradation,
                "enhancement": enhancement,
                "normalization": normalization,
                "wer": wer_value,
                "time_s": res["transcription"]["processing_time"],
                "transcription": res["transcription"],
                "reference_text": res.get("reference_text", ""),
            })
        return prepared
