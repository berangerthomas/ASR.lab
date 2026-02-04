import click
import json
from pathlib import Path

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
    import sys
    config_name = config_path.stem
    
    try:
        from ..benchmarks.runner import BenchmarkRunner
        runner = BenchmarkRunner(config_path)
        runner.run()
        
        # Check for partial failures
        if runner.failed_engines:
            click.echo(f"\n⚠️  Benchmark completed with warnings!", err=True)
            click.echo(f"   Failed engines: {', '.join(runner.failed_engines)}", err=True)
        else:
            click.echo("\n✅ Benchmark finished successfully!")
        
        click.echo(f"\n📊 Report available at: results/reports/{config_name}/report_interactive.html")
        
    except RuntimeError as e:
        click.echo(f"\n❌ Benchmark failed: {e}", err=True)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"\n❌ Configuration error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option(
    '--results', '-r',
    'results_path',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to raw_results.json file or directory containing it."
)
def report(results_path: Path):
    """
    Regenerate the interactive report from existing results.
    
    Examples:
        python main.py report -r results/reports/quick_test/raw_results.json
        python main.py report -r results/reports/quick_test
    """
    try:
        # Determine results path and output directory
        if results_path:
            # Check if it's a directory or file
            if results_path.is_dir():
                reports_dir = results_path
                results_path = reports_dir / 'raw_results.json'
            else:
                reports_dir = results_path.parent
        else:
            # List available configs with results
            reports_base = Path('results/reports')
            available_configs = []
            if reports_base.exists():
                for subdir in reports_base.iterdir():
                    if subdir.is_dir() and (subdir / 'raw_results.json').exists():
                        available_configs.append(subdir.name)
            
            if available_configs:
                click.echo("Available benchmark results:")
                for cfg in sorted(available_configs):
                    click.echo(f"  - results/reports/{cfg}/raw_results.json")
                click.echo("\nUsage: python main.py report -r <path>")
                click.echo(f"Example: python main.py report -r results/reports/{available_configs[0]}")
            else:
                click.echo("Error: No benchmark results found.", err=True)
                click.echo("Please run a benchmark first using: python main.py run -c <config>", err=True)
            return
        
        if not results_path.exists():
            click.echo(f"Error: Results file not found at '{results_path}'.", err=True)
            click.echo("Please run a benchmark first using the 'run' command.", err=True)
            return
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            click.echo("Warning: The results file is empty. No report will be generated.", err=True)
            return

        from ..analysis.report_interactive import InteractiveReportGenerator
        report_generator = InteractiveReportGenerator(results, reports_dir)
        report_generator.generate_report()
        
        click.echo("\n✅ Report regenerated successfully!")
        click.echo(f"\n📊 Report available at: {reports_dir / 'report_interactive.html'}")

    except json.JSONDecodeError:
        click.echo(f"Error: Could not decode JSON from '{results_path}'. The file might be corrupted.", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
