import click
import json
from pathlib import Path

from ..benchmarks.runner import BenchmarkRunner
from ..analysis.report_interactive import InteractiveReportGenerator

@click.group()
def cli():
    """ASR.lab - A benchmarking tool for ASR engines."""
    pass

@cli.command()
@click.option(
    '--config', '-c',
    'config_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default='configs/default.yaml',
    help="Path to the benchmark configuration YAML file. Defaults to 'configs/default.yaml'."
)
def run(config_path: Path):
    """
    Run a benchmark using the specified configuration file.
    """
    try:
        runner = BenchmarkRunner(config_path)
        runner.run()
        click.echo("\n✅ Benchmark finished successfully!")
        click.echo("\n📊 Report available at: results/reports/report_interactive.html")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.option(
    '--results', '-r',
    'results_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default='results/reports/raw_results.json',
    help="Path to the raw results JSON file. Defaults to 'results/reports/raw_results.json'."
)
def report(results_path: Path):
    """
    Regenerate the interactive report from existing results.
    """
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            click.echo("Warning: The results file is empty. No report will be generated.", err=True)
            return

        reports_dir = results_path.parent
        report_generator = InteractiveReportGenerator(results, reports_dir)
        report_generator.generate_report()
        
        click.echo("\n✅ Report regenerated successfully!")
        click.echo(f"\n📊 Report available at: {reports_dir / 'report_interactive.html'}")

    except FileNotFoundError:
        click.echo(f"Error: Results file not found at '{results_path}'.", err=True)
        click.echo("Please run a benchmark first using the 'run' command.", err=True)
    except json.JSONDecodeError:
        click.echo(f"Error: Could not decode JSON from '{results_path}'. The file might be corrupted.", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
