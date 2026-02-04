import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Any

class PlotlyVisualizer:
    """
    Generates interactive visualizations from benchmark results using Plotly.
    """

    def __init__(self, results: List[Dict[str, Any]], output_dir: Path):
        self.df = self._prepare_dataframe(results)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def _prepare_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts the results list into a structured pandas DataFrame."""
        records = []
        for res in results:
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

            record = {
                "language": language,
                "engine": res["engine"],
                "dataset": res["dataset"],
                "degradation": degradation,
                "enhancement": enhancement,
                "normalization": normalization,
                "wer": wer_value,
                "time_s": res["transcription"]["processing_time"],
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df.dropna(subset=['wer', 'time_s'], inplace=True)
        return df

    def create_interactive_plot(self):
        """Creates a single interactive HTML file with all visualizations and filters."""
        if self.df.empty:
            print("No data available for plotting.")
            return

        languages = self.df['language'].unique()
        engines = self.df['engine'].unique()
        degradations = self.df['degradation'].unique()

        enhancements = self.df['enhancement'].unique()
        normalizations = self.df['normalization'].unique()

        # Create subplots - one per language
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

        # Color mapping for engines
        color_map = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for i, engine in enumerate(engines):
            color_map[engine] = colors[i % len(colors)]

        # Symbol mapping for degradation types
        symbol_map = {
            'original': 'circle',
            # Other degradations will use different symbols
        }
        available_symbols = ['diamond', 'square', 'cross', 'x', 'triangle-up', 'star']
        symbol_idx = 0
        for deg in degradations:
            if deg not in symbol_map:
                symbol_map[deg] = available_symbols[symbol_idx % len(available_symbols)]
                symbol_idx += 1

        # Add traces for each combination of language, engine, degradation, enhancement, normalization
        for lang_idx, lang in enumerate(languages):
            row = (lang_idx // cols) + 1
            col = (lang_idx % cols) + 1
            
            lang_df = self.df[self.df['language'] == lang]
            
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
                            
                            enhancement_label = f" + {enhancement}" if enhancement != "None" else ""
                            normalization_label = f" + {normalization}" if normalization != "None" else ""

                            # Create hover text with details
                            hover_text = [
                                f"<b>{engine}</b><br>" +
                                f"WER: {wer:.4f}<br>" +
                                f"Time: {time_s:.2f}s<br>" +
                                f"Type: {degradation}<br>" +
                                f"Enhancement: {enhancement}<br>" +
                                f"Normalization: {normalization}<br>" +
                                f"Dataset: {dataset}"
                                for wer, time_s, dataset in zip(
                                    subset['wer'], 
                                    subset['time_s'],
                                    subset['dataset']
                                )
                            ]
                            
                            # Show legend only for the first subplot
                            show_legend = (lang_idx == 0)
                            
                            # Legend group to sync visibility across subplots
                            legend_group = f"{engine}_{degradation}_{enhancement}_{normalization}"
                            
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
                                        line=dict(width=1, color='white')
                                    ),
                                    hovertext=hover_text,
                                    hoverinfo='text',
                                ),
                                row=row,
                                col=col
                            )

        # Update axes labels
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(title_text="Processing Time (s)", row=i, col=j)
                fig.update_yaxes(title_text="Word Error Rate (WER)", row=i, col=j)

        # Create filter buttons
        buttons_engine = []
        buttons_degradation = []

        # Button: Show All
        buttons_engine.append(
            dict(
                label="All Engines",
                method="update",
                args=[{"visible": [True] * len(fig.data)}]
            )
        )

        # Buttons: Filter by engine
        for engine in engines:
            visibility = []
            for trace in fig.data:
                visibility.append(engine in trace.name)
            
            buttons_engine.append(
                dict(
                    label=engine,
                    method="update",
                    args=[{"visible": visibility}]
                )
            )

        # Button: Show All degradations
        buttons_degradation.append(
            dict(
                label="All Types",
                method="update",
                args=[{"visible": [True] * len(fig.data)}]
            )
        )

        # Buttons: Filter by degradation
        for deg in degradations:
            visibility = []
            for trace in fig.data:
                visibility.append(f"({deg})" in trace.name)
            
            buttons_degradation.append(
                dict(
                    label=deg.capitalize(),
                    method="update",
                    args=[{"visible": visibility}]
                )
            )

        # Update layout with buttons
        fig.update_layout(
            title={
                'text': "ASR Benchmark Results - Interactive Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=400 * rows,
            hovermode='closest',
            template='plotly_white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10)
            ),
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.02,
                    y=1.15,
                    buttons=buttons_engine,
                    showactive=True,
                    bgcolor="lightgray",
                    bordercolor="gray",
                    font=dict(size=11),
                ),
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.18,
                    y=1.15,
                    buttons=buttons_degradation,
                    showactive=True,
                    bgcolor="lightgray",
                    bordercolor="gray",
                    font=dict(size=11),
                ),
            ],
            annotations=[
                dict(
                    text="Filter by Engine:",
                    x=0.02,
                    y=1.18,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xanchor="left"
                ),
                dict(
                    text="Filter by Type:",
                    x=0.18,
                    y=1.18,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xanchor="left"
                ),
            ]
        )

        # Save as standalone HTML
        output_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(
            str(output_path),
            include_plotlyjs=True,  # Fully autonomous - no internet required
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'asr_benchmark_plot',
                    'height': 1000,
                    'width': 1400,
                    'scale': 2
                }
            }
        )
        print(f"Interactive dashboard saved to {output_path}")

    def run_visualizations(self):
        """Runs all visualization methods."""
        self.create_interactive_plot()
