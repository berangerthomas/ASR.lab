import jinja2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Any
import datetime
from jiwer.process import process_words, process_characters

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
        # Generate the Plotly chart
        plotly_div = self._create_plotly_chart()
        
        # Prepare data for the template
        prepared_results = self._prepare_results_for_template()
        
        # Extract unique engines, languages, degradations, enhancements, and normalizations
        engines = sorted(set(r['engine'] for r in prepared_results))
        languages = sorted(set(r['language'] for r in prepared_results))
        degradations = sorted(set(r['degradation'] for r in prepared_results))
        enhancements = sorted(set(r['enhancement'] for r in prepared_results))
        normalizations = sorted(set(r.get('normalization', 'None') for r in prepared_results))
        
        template_data = {
            "generation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": prepared_results,
            "plotly_div": plotly_div,
            "engines": engines,
            "languages": languages,
            "degradations": degradations,
            "enhancements": enhancements,
            "normalizations": normalizations,
            "metrics_info": self.metrics_info,
        }

        # Load and render the template
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_dir))
        template = env.get_template("report_interactive.html")
        html_content = template.render(template_data)

        # Save the report
        report_path = self.output_dir / "report_interactive.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive report saved to {report_path}")
        
        # Print file size
        file_size_mb = report_path.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

    def _create_plotly_chart(self) -> str:
        """Creates an interactive Plotly chart and returns the HTML div."""
        import pandas as pd
        
        # Prepare dataframe
        records = []
        for res in self.results:
            # Use fields from result if available, else fallback to parsing
            language = res.get("language") or res["dataset"].split('_')[0]
            degradation = res.get("degradation")
            enhancement = res.get("enhancement", "None")
            
            if not degradation:
                is_pristine = "original" in res["dataset"]
                degradation = "original" if is_pristine else res["dataset"].split("degraded_")[-1]

            # Get all metrics
            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")
            cer_value = res["metrics"].get("cer") or res["metrics"].get("CER")
            mer_value = res["metrics"].get("mer") or res["metrics"].get("MER")
            wil_value = res["metrics"].get("wil") or res["metrics"].get("WIL")
            wip_value = res["metrics"].get("wip") or res["metrics"].get("WIP")
            
            normalization = res.get("normalization")
            if not normalization or normalization == "None":
                # Fallback: try to parse from dataset name
                dataset_name = res["dataset"]
                if dataset_name.endswith("norm_minus_18"):
                    normalization = "norm_minus_18"
                elif dataset_name.endswith("no_norm"):
                    normalization = "no_norm"
                else:
                    normalization = "None"

            records.append({
                "language": language,
                "engine": res["engine"],
                "dataset": res["dataset"],
                "degradation": degradation,
                "enhancement": enhancement,
                "normalization": normalization,
                "wer": wer_value,
                "cer": cer_value,
                "mer": mer_value,
                "wil": wil_value,
                "wip": wip_value,
                "time_s": res["transcription"]["processing_time"],
            })
        
        df = pd.DataFrame(records)
        df.dropna(subset=['wer', 'time_s'], inplace=True)
        
        if df.empty:
            return "<p>No data available for plotting.</p>"

        languages = df['language'].unique()
        engines = df['engine'].unique()
        degradations = df['degradation'].unique()
        enhancements = df['enhancement'].unique()
        normalizations = df['normalization'].unique()

        # Create subplots
        num_langs = len(languages)
        rows = (num_langs + 1) // 2
        cols = 2 if num_langs > 1 else 1

        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f"Language: {lang.upper()}" for lang in languages],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Color and symbol mapping
        color_map = {}
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#feca57', '#ff6b6b']
        for i, engine in enumerate(engines):
            color_map[engine] = colors[i % len(colors)]

        symbol_map = {'original': 'circle', 'None': 'circle'}
        available_symbols = ['diamond', 'square', 'cross', 'x', 'triangle-up', 'star']
        symbol_idx = 0
        for deg in degradations:
            if deg not in symbol_map:
                symbol_map[deg] = available_symbols[symbol_idx % len(available_symbols)]
                symbol_idx += 1

        # Add traces
        for lang_idx, lang in enumerate(languages):
            row = (lang_idx // cols) + 1
            col = (lang_idx % cols) + 1
            
            lang_df = df[df['language'] == lang]
            
            for engine in engines:
                for degradation in degradations:
                    for enhancement in enhancements:
                        for normalization in normalizations:
                            mask = (lang_df['engine'] == engine) & \
                                   (lang_df['degradation'] == degradation) & \
                                   (lang_df['enhancement'] == enhancement) & \
                                   (lang_df['normalization'] == normalization)
                            subset = lang_df[mask]
                            
                            if subset.empty:
                                continue
                            
                            # Differentiate enhancement in legend/hover
                            enhancement_label = f" + {enhancement}" if enhancement != "None" else ""
                            normalization_label = f" + {normalization}" if normalization != "None" else ""
                            
                            # Helper function to format metric values (handles None and NaN)
                            def fmt_metric(val, precision=4):
                                if val is None:
                                    return "N/A"
                                try:
                                    import pandas as pd
                                    if pd.isna(val):
                                        return "N/A"
                                    return f"{float(val):.{precision}f}"
                                except (ValueError, TypeError):
                                    return "N/A"
                            
                            # Safe column access for optional metrics
                            cer_vals = subset['cer'] if 'cer' in subset.columns else [None] * len(subset)
                            mer_vals = subset['mer'] if 'mer' in subset.columns else [None] * len(subset)
                            wil_vals = subset['wil'] if 'wil' in subset.columns else [None] * len(subset)
                            wip_vals = subset['wip'] if 'wip' in subset.columns else [None] * len(subset)
                            
                            hover_text = [
                                f"<b>{engine}</b><br>" +
                                f"WER: {fmt_metric(wer)}<br>" +
                                f"CER: {fmt_metric(cer)}<br>" +
                                f"MER: {fmt_metric(mer)}<br>" +
                                f"WIL: {fmt_metric(wil)}<br>" +
                                f"WIP: {fmt_metric(wip)}<br>" +
                                f"Time: {fmt_metric(time_s, 2)}s<br>" +
                                f"Degradation: {degradation}<br>" +
                                f"Enhancement: {enhancement}<br>" +
                                f"Normalization: {normalization}<br>" +
                                f"Dataset: {dataset}"
                                for wer, cer, mer, wil, wip, time_s, dataset in zip(
                                    subset['wer'], 
                                    cer_vals,
                                    mer_vals,
                                    wil_vals,
                                    wip_vals,
                                    subset['time_s'],
                                    subset['dataset']
                                )
                            ]
                            
                            show_legend = (lang_idx == 0)
                            legend_group = f"{engine}_{degradation}_{enhancement}_{normalization}"
                            
                            # Prepare customdata for robust filtering in JS
                            # Format: [engine, degradation, enhancement, normalization, dataset, wer, cer, mer, wil, wip]
                            # Use .get() with default for missing columns
                            cer_col = subset['cer'] if 'cer' in subset.columns else [None] * len(subset)
                            mer_col = subset['mer'] if 'mer' in subset.columns else [None] * len(subset)
                            wil_col = subset['wil'] if 'wil' in subset.columns else [None] * len(subset)
                            wip_col = subset['wip'] if 'wip' in subset.columns else [None] * len(subset)
                            
                            custom_data = [
                                [engine, degradation, enhancement, normalization, dataset,
                                 wer_val, cer_val, mer_val, wil_val, wip_val]
                                for dataset, wer_val, cer_val, mer_val, wil_val, wip_val in zip(
                                    subset['dataset'],
                                    subset['wer'],
                                    cer_col,
                                    mer_col,
                                    wil_col,
                                    wip_col
                                )
                            ]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=subset['time_s'],
                                    y=subset['wer'],
                                    mode='markers',
                                    name=f"{engine} ({degradation}{enhancement_label}{normalization_label})",
                                    legendgroup=legend_group,
                                    showlegend=show_legend,
                                    marker=dict(
                                        size=12,
                                        color=color_map[engine],
                                        symbol=symbol_map[degradation],
                                        line=dict(width=1, color='black' if enhancement != "None" else 'white'),
                                        opacity=0.8 if enhancement != "None" else 1.0
                                    ),
                                    hovertext=hover_text,
                                    hoverinfo='text',
                                    customdata=custom_data,
                                ),
                                row=row,
                                col=col
                            )

        # Update axes
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(title_text="Processing Time (s)", row=i, col=j)
                fig.update_yaxes(title_text="Word Error Rate (WER)", row=i, col=j)

        # Update layout
        fig.update_layout(
            height=400 * rows,
            hovermode='closest',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10)
            ),
            margin=dict(t=100, b=50, l=50, r=150),
        )

        # Convert to HTML div (fully autonomous with embedded Plotly.js)
        plotly_html = fig.to_html(
            include_plotlyjs=True,
            full_html=False,
            div_id='plotly-chart',
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'asr_benchmark',
                    'height': 1000,
                    'width': 1400,
                    'scale': 2
                }
            }
        )
        
        return plotly_html

    def _prepare_results_for_template(self) -> List[Dict[str, Any]]:
        """Prepares the results data for easy rendering in the template."""
        prepared = []
        for res in self.results:
            language = res.get("language") or res["dataset"].split('_')[0]
            degradation = res.get("degradation")
            enhancement = res.get("enhancement", "None")
            normalization = res.get("normalization")
            if not normalization or normalization == "None":
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

            # Get all metrics
            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")
            cer_value = res["metrics"].get("cer") or res["metrics"].get("CER")
            mer_value = res["metrics"].get("mer") or res["metrics"].get("MER")
            wil_value = res["metrics"].get("wil") or res["metrics"].get("WIL")
            wip_value = res["metrics"].get("wip") or res["metrics"].get("WIP")

            ref_text = res.get("reference_text", "")
            trans_text = res["transcription"]["text"]

            # Generate word-level alignment (for WER, MER, WIL, WIP)
            word_ref_html, word_trans_html = self._generate_word_alignment(ref_text, trans_text)
            
            # Generate character-level alignment (for CER)
            char_ref_html, char_trans_html = self._generate_char_alignment(ref_text, trans_text)

            prepared.append({
                "dataset": res["dataset"],
                "engine": res["engine"],
                "language": language,
                "degradation": degradation,
                "enhancement": enhancement,
                "normalization": normalization,
                "wer": wer_value,
                "cer": cer_value,
                "mer": mer_value,
                "wil": wil_value,
                "wip": wip_value,
                "time_s": res["transcription"]["processing_time"],
                "transcription": res["transcription"],
                # Word-level diff (for WER/MER/WIL/WIP)
                "reference_text_words": word_ref_html,
                "transcribed_text_words": word_trans_html,
                # Character-level diff (for CER)
                "reference_text_chars": char_ref_html,
                "transcribed_text_chars": char_trans_html,
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

    def _generate_char_alignment(self, ref_text: str, trans_text: str) -> tuple[str, str]:
        """Generate HTML with character-level alignment highlighting."""
        try:
            output = process_characters(ref_text, trans_text)
            ref_html, trans_html = [], []

            for chunk in output.alignments[0]:
                ref_chars = output.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
                hyp_chars = output.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]

                if chunk.type == 'equal':
                    ref_html.append(''.join(ref_chars))
                    trans_html.append(''.join(hyp_chars))
                elif chunk.type == 'substitute':
                    ref_html.append(f'<span class="diff-del">{"".join(ref_chars)}</span>')
                    trans_html.append(f'<span class="diff-add">{"".join(hyp_chars)}</span>')
                elif chunk.type == 'delete':
                    ref_html.append(f'<span class="diff-del">{"".join(ref_chars)}</span>')
                elif chunk.type == 'insert':
                    trans_html.append(f'<span class="diff-add">{"".join(hyp_chars)}</span>')

            return "".join(ref_html), "".join(trans_html)
        except Exception:
            return ref_text, trans_text
