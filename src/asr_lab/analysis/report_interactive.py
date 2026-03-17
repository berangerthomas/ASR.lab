import json
import jinja2
from pathlib import Path
from typing import List, Dict, Any
import datetime
from jiwer.process import process_words


class InteractiveReportGenerator:
    """
    Generates a complete interactive HTML report with Plotly charts and data tables.
    """

    def __init__(self, results: List[Dict[str, Any]], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.template_dir = Path(__file__).parent / "templates"
        # Available metrics with display names and descriptions
        self.metrics_info = {
            "wer": {
                "name": "Word Error Rate (WER)",
                "lower_better": True,
                "description": "WER = (Substitutions + Deletions + Insertions) / Reference Words. Most common ASR metric. Lower is better."
            },
            "cer": {
                "name": "Character Error Rate (CER)",
                "lower_better": True,
                "description": "CER = Character-level edit distance / Reference Characters. More granular than WER. Lower is better."
            },
            "mer": {
                "name": "Match Error Rate (MER)",
                "lower_better": True,
                "description": "MER = Errors / (Hits + Errors). Bounded 0-1, accounts for hypothesis length. Lower is better."
            },
            "wil": {
                "name": "Word Information Lost (WIL)",
                "lower_better": True,
                "description": "WIL = 1 - WIP. Measures proportion of word information lost. Lower is better."
            },
            "wip": {
                "name": "Word Information Preserved (WIP)",
                "lower_better": False,
                "description": "WIP = Hits² / (Ref Length × Hyp Length). Measures preserved information. Higher is better."
            },
        }

    def generate_report(self):
        """
        Renders a complete interactive HTML report with embedded Plotly charts.
        """
        # Prepare chart data as JSON for client-side rendering
        chart_data = self._prepare_chart_data()
        
        # Prepare data for the template
        prepared_results = self._prepare_results_for_template()
        
        # Extract unique engines, languages, degradations, enhancements, and normalizations
        engines = sorted(set(r['engine'] for r in prepared_results))
        languages = sorted(set(r['language'] for r in prepared_results))
        degradations = sorted(set(r['degradation'] for r in prepared_results))
        enhancements = sorted(set(r['enhancement'] for r in prepared_results))
        audio_norms = sorted(set(r.get('audio_norm', 'None') for r in prepared_results))
        
        template_data = {
            "generation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": prepared_results,
            "chart_data_json": json.dumps(chart_data, default=str),
            "engines": engines,
            "languages": languages,
            "degradations": degradations,
            "enhancements": enhancements,
            "audio_norms": audio_norms,
            "metrics_info": self.metrics_info,
        }

        # Load and render the template
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_dir))
        template = env.get_template("report_interactive.html")
        html_content = template.render(template_data)

        # Save the report
        report_filename = f"report_{self.output_dir.name}.html"
        report_path = self.output_dir / report_filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive report saved to {report_path}")
        
        # Print file size
        file_size_mb = report_path.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

    def _prepare_chart_data(self) -> list:
        """Prepares result data as a JSON-serializable list for client-side Plotly rendering."""
        records = []
        for idx, res in enumerate(self.results):
            language = res.get("language") or res["dataset"].split('_')[0]
            degradation = res.get("degradation")
            enhancement = res.get("enhancement", "None")

            if not degradation:
                is_pristine = "original" in res["dataset"]
                degradation = "original" if is_pristine else res["dataset"].split("degraded_")[-1]

            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")
            cer_value = res["metrics"].get("cer") or res["metrics"].get("CER")
            mer_value = res["metrics"].get("mer") or res["metrics"].get("MER")
            wil_value = res["metrics"].get("wil") or res["metrics"].get("WIL")
            wip_value = res["metrics"].get("wip") or res["metrics"].get("WIP")

            audio_norm = res.get("audio_norm") or res.get("normalization")
            if not audio_norm or audio_norm == "None":
                dataset_name = res["dataset"]
                if dataset_name.endswith("norm_minus_18"):
                    audio_norm = "norm_minus_18"
                elif dataset_name.endswith("no_norm"):
                    audio_norm = "no_norm"
                else:
                    audio_norm = "None"

            text_norm = res.get("text_norm", "raw")
            time_s = res["transcription"]["processing_time"]

            # Skip records with missing essential data
            if wer_value is None or time_s is None:
                continue

            records.append({
                "idx": idx,
                "lang": language,
                "engine": res["engine"],
                "dataset": res["dataset"],
                "degradation": degradation,
                "enhancement": enhancement,
                "audio_norm": audio_norm,
                "text_norm": text_norm,
                "wer": wer_value,
                "cer": cer_value,
                "mer": mer_value,
                "wil": wil_value,
                "wip": wip_value,
                "time_s": time_s,
            })
        return records

    def _prepare_results_for_template(self) -> List[Dict[str, Any]]:
        """Prepares the results data for easy rendering in the template."""
        prepared = []
        for res in self.results:
            language = res.get("language") or res["dataset"].split('_')[0]
            degradation = res.get("degradation")
            enhancement = res.get("enhancement", "None")
            
            # Audio normalization (distinct from text normalization)
            audio_norm = res.get("audio_norm") or res.get("normalization")
            if not audio_norm or audio_norm == "None":
                # Fallback: try to parse from dataset name
                dataset_name = res["dataset"]
                if dataset_name.endswith("norm_minus_18"):
                    audio_norm = "norm_minus_18"
                elif dataset_name.endswith("no_norm"):
                    audio_norm = "no_norm"
                else:
                    audio_norm = "None"
            
            # Text normalization preset
            text_norm = res.get("text_norm", "raw")
            text_norm_display = res.get("text_norm_display", "Raw (none)")
            
            if not degradation:
                is_pristine = "original" in res["dataset"]
                degradation = "original" if is_pristine else res["dataset"].split("degraded_")[-1]

            # Get all metrics
            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")
            cer_value = res["metrics"].get("cer") or res["metrics"].get("CER")
            mer_value = res["metrics"].get("mer") or res["metrics"].get("MER")
            wil_value = res["metrics"].get("wil") or res["metrics"].get("WIL")
            wip_value = res["metrics"].get("wip") or res["metrics"].get("WIP")

            # Use normalized texts if available, else raw
            ref_text = res.get("reference_normalized") or res.get("reference_text", "")
            trans_text = res.get("hypothesis_normalized") or res["transcription"]["text"]

            # Generate word-level alignment (for WER, MER, WIL, WIP)
            word_ref_html, word_trans_html = self._generate_word_alignment(ref_text, trans_text)
            
            prepared.append({
                "dataset": res["dataset"],
                "engine": res["engine"],
                "language": language,
                "degradation": degradation,
                "enhancement": enhancement,
                "audio_norm": audio_norm,
                "text_norm": text_norm,
                "text_norm_display": text_norm_display,
                "wer": wer_value,
                "cer": cer_value,
                "mer": mer_value,
                "wil": wil_value,
                "wip": wip_value,
                "time_s": res["transcription"]["processing_time"],
                # Word-level diff (for WER/MER/WIL/WIP)
                "reference_text_words": word_ref_html,
                "transcribed_text_words": word_trans_html,
            })
        return prepared

    def _generate_word_alignment(self, ref_text: str, trans_text: str) -> tuple[str, str]:
        """Generate HTML with word-level alignment highlighting."""
        try:
            output = process_words(ref_text, trans_text)
            ref_html, trans_html = [], []

            for chunk in output.alignments[0]:
                ref_words = output.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
                hyp_words = output.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]

                if chunk.type == 'equal':
                    ref_html.append(' '.join(ref_words))
                    trans_html.append(' '.join(hyp_words))
                elif chunk.type == 'substitute':
                    ref_html.append(f'<span class="diff-del">{" ".join(ref_words)}</span>')
                    trans_html.append(f'<span class="diff-add">{" ".join(hyp_words)}</span>')
                elif chunk.type == 'delete':
                    ref_html.append(f'<span class="diff-del">{" ".join(ref_words)}</span>')
                elif chunk.type == 'insert':
                    trans_html.append(f'<span class="diff-add">{" ".join(hyp_words)}</span>')

            return " ".join(ref_html), " ".join(trans_html)
        except Exception:
            return ref_text, trans_text


