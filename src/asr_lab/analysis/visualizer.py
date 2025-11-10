import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

class Visualizer:
    """
    Generates visualizations from benchmark results.
    """

    def __init__(self, results: List[Dict[str, Any]], output_dir: Path):
        self.df = self._prepare_dataframe(results)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _prepare_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts the results list into a structured pandas DataFrame."""
        records = []
        for res in results:
            language = res["dataset"].split('_')[0]
            is_pristine = "original" in res["dataset"]
            degradation = "original" if is_pristine else res["dataset"].split("degraded_")[-1]

            # Get WER value (try both lowercase and uppercase for backward compatibility)
            wer_value = res["metrics"].get("wer") or res["metrics"].get("WER")

            record = {
                "language": language,
                "engine": res["engine"],
                "dataset": res["dataset"],
                "degradation": degradation,
                "wer": wer_value,
                "time_s": res["transcription"]["processing_time"],
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df.dropna(subset=['wer', 'time_s'], inplace=True)
        return df

    def plot_wer_vs_time_original(self):
        """Generates a scatter plot of WER vs. Time for original audio."""
        self._plot_wer_vs_time(self.df[self.df['degradation'] == 'original'], "Original Audio", "wer_vs_time_original.png")

    def plot_wer_vs_time_degraded(self):
        """Generates a scatter plot of WER vs. Time for degraded audio."""
        self._plot_wer_vs_time(self.df[self.df['degradation'] != 'original'], "Degraded Audio", "wer_vs_time_degraded.png")

    def plot_wer_vs_time_combined(self):
        """Generates a scatter plot of WER vs. Time for all audio, with style differentiation."""
        self._plot_wer_vs_time(self.df, "Combined Audio", "wer_vs_time_combined.png", use_style=True)

    def _plot_wer_vs_time(self, df: pd.DataFrame, title_prefix: str, filename: str, use_style: bool = False):
        """Helper function to generate scatter plots."""
        if df.empty:
            print(f"No data available for {title_prefix} plot.")
            return

        languages = df['language'].unique()
        num_langs = len(languages)
        
        if num_langs == 0:
            return

        fig, axes = plt.subplots(
            nrows=(num_langs + 1) // 2, ncols=2,
            figsize=(15, 5 * ((num_langs + 1) // 2)),
            squeeze=False
        )
        axes = axes.flatten()

        for idx, lang in enumerate(languages):
            ax = axes[idx]
            lang_df = df[df['language'] == lang]
            
            plot_params = {
                "data": lang_df,
                "x": "time_s",
                "y": "wer",
                "hue": "engine",
                "ax": ax,
                "s": 100
            }
            if use_style:
                plot_params["style"] = "degradation"

            sns.scatterplot(**plot_params)
            
            for _, row in lang_df.iterrows():
                ax.text(row['time_s'] + 0.1, row['wer'], row['engine'], fontsize=9)

            ax.set_title(f"Performance on {title_prefix} - Language: {lang.upper()}")
            ax.set_xlabel("Processing Time (s)")
            ax.set_ylabel("Word Error Rate (WER)")
            ax.legend().set_visible(True)

        for j in range(num_langs, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

    def run_visualizations(self):
        """Runs all visualization methods."""
        self.plot_wer_vs_time_original()
        self.plot_wer_vs_time_degraded()
        self.plot_wer_vs_time_combined()
