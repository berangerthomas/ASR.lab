import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

class CsvExporter:
    """
    Exports benchmark results to a CSV file.
    """

    def __init__(self, results: List[Dict[str, Any]], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def export(self, filename: str = "results.csv"):
        """
        Saves the results to a CSV file.
        """
        df = self._prepare_dataframe()
        save_path = self.output_dir / filename
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"Results exported to {save_path}")

    def _prepare_dataframe(self) -> pd.DataFrame:
        """Converts the results list into a structured pandas DataFrame for export."""
        records = []
        for res in self.results:
            language = res["dataset"].split('_')[0]
            is_pristine = "original" in res["dataset"]
            degradation = "original" if is_pristine else res["dataset"].split("degraded_")[-1]

            # Get metric values (try both lowercase and uppercase for backward compatibility)
            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")
            rtf_value = res["metrics"].get("rtf") or res["metrics"].get("RTF")
            cer_value = res["metrics"].get("cer") or res["metrics"].get("CER")

            record = {
                "language": language,
                "engine": res["engine"],
                "dataset": res["dataset"],
                "degradation": degradation,
                "wer": wer_value,
                "rtf": rtf_value,
                "cer": cer_value,
                "processing_time_s": res["transcription"]["processing_time"],
                "reference_text": res.get("reference_text", ""),
                "transcribed_text": res["transcription"]["text"],
            }
            # Add all metrics to the record
            for metric_name, metric_value in res["metrics"].items():
                record[f"metric_{metric_name}"] = metric_value

            records.append(record)
        
        return pd.DataFrame(records)
