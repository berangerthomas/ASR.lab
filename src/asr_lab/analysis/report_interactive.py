import jinja2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    def generate_report(self):
        """
        Renders a complete interactive HTML report with embedded Plotly charts.
        """
        # Generate the Plotly chart
        plotly_div = self._create_plotly_chart()
        
        # Prepare data for the template
        prepared_results = self._prepare_results_for_template()
        
        # Extract unique engines, languages, and degradations
        engines = sorted(set(r['engine'] for r in prepared_results))
        languages = sorted(set(r['language'] for r in prepared_results))
        degradations = sorted(set(r['degradation'] for r in prepared_results))
        
        template_data = {
            "generation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": prepared_results,
            "plotly_div": plotly_div,
            "engines": engines,
            "languages": languages,
            "degradations": degradations,
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
            language = res["dataset"].split('_')[0]
            is_pristine = "original" in res["dataset"]
            degradation = "original" if is_pristine else res["dataset"].split("degraded_")[-1]
            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")

            records.append({
                "language": language,
                "engine": res["engine"],
                "dataset": res["dataset"],
                "degradation": degradation,
                "wer": wer_value,
                "time_s": res["transcription"]["processing_time"],
            })
        
        df = pd.DataFrame(records)
        df.dropna(subset=['wer', 'time_s'], inplace=True)
        
        if df.empty:
            return "<p>No data available for plotting.</p>"

        languages = df['language'].unique()
        engines = df['engine'].unique()
        degradations = df['degradation'].unique()

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

        symbol_map = {'original': 'circle'}
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
                    mask = (lang_df['engine'] == engine) & (lang_df['degradation'] == degradation)
                    subset = lang_df[mask]
                    
                    if subset.empty:
                        continue
                    
                    hover_text = [
                        f"<b>{engine}</b><br>" +
                        f"WER: {wer:.4f}<br>" +
                        f"Time: {time_s:.2f}s<br>" +
                        f"Type: {degradation}<br>" +
                        f"Dataset: {dataset}"
                        for wer, time_s, dataset in zip(
                            subset['wer'], 
                            subset['time_s'],
                            subset['dataset']
                        )
                    ]
                    
                    show_legend = (lang_idx == 0)
                    legend_group = f"{engine}_{degradation}"
                    
                    fig.add_trace(
                        go.Scatter(
                            x=subset['time_s'],
                            y=subset['wer'],
                            mode='markers',
                            name=f"{engine} ({degradation})",
                            legendgroup=legend_group,
                            showlegend=show_legend,
                            marker=dict(
                                size=12,
                                color=color_map[engine],
                                symbol=symbol_map[degradation],
                                line=dict(width=1, color='white')
                            ),
                            hovertext=hover_text,
                            hoverinfo='text',
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
            language = res["dataset"].split('_')[0]
            is_pristine = "original" in res["dataset"]
            degradation = "original" if is_pristine else res["dataset"].split("degraded_")[-1]
            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")

            ref_text = res.get("reference_text", "")
            trans_text = res["transcription"]["text"]

            # Use jiwer to get the alignment corresponding to the WER calculation
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

            prepared.append({
                "dataset": res["dataset"],
                "engine": res["engine"],
                "language": language,
                "degradation": degradation,
                "wer": wer_value,
                "time_s": res["transcription"]["processing_time"],
                "transcription": res["transcription"], # Keep original transcription for other uses if needed
                "reference_text": " ".join(ref_html),
                "transcribed_text": " ".join(trans_html),
            })
        return prepared
