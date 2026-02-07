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
    
    Metrics are pre-computed for all normalization variants (8 combinations)
    and stored in the JSON results. The interactive HTML report allows
    switching between normalization options via checkboxes.
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
    help="Path to raw_results.json or directory containing it."
)
def report(results_path: Path):
    """
    Regenerate an interactive report from existing benchmark results.
    
    The interactive HTML report includes checkboxes to switch between
    normalization options (lowercase, punctuation, contractions) in real-time.
    All metric variants are pre-computed using jiwer.
    
    Examples:
    
        # Regenerate report from a results directory
        python main.py report -r results/reports/demo
        
        # Regenerate report from a specific JSON file
        python main.py report -r results/reports/demo/raw_results.json
    """
    from ..analysis.report_interactive import InteractiveReportGenerator
    
    try:
        # Determine paths
        if results_path:
            if results_path.is_dir():
                reports_dir = results_path
                json_path = reports_dir / 'raw_results.json'
            else:  # Assume JSON
                json_path = results_path
                reports_dir = results_path.parent
        else:
            # List available results
            _list_available_results()
            return
        
        if not json_path.exists():
            click.echo(f"❌ No results found at {json_path}", err=True)
            return
        
        click.echo(f"📂 Loading results from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            click.echo("❌ No results found in file.", err=True)
            return
        
        click.echo(f"📊 Generating report for {len(results)} transcriptions...")
        
        # Generate report
        report_generator = InteractiveReportGenerator(results, reports_dir)
        report_generator.generate_report()
        
        click.echo("\n✅ Report generated successfully!")
        click.echo(f"\n📊 Report available at: {reports_dir / 'report_interactive.html'}")
        click.echo("\n💡 Use checkboxes in the report to switch normalization options.")

    except Exception as e:
        click.echo(f"❌ Error generating report: {e}", err=True)
        import traceback
        traceback.print_exc()


def _list_available_results():
    """List available benchmark results."""
    reports_base = Path('results/reports')
    available = []
    
    if reports_base.exists():
        for subdir in reports_base.iterdir():
            if subdir.is_dir():
                has_json = (subdir / 'raw_results.json').exists()
                if has_json:
                    available.append(subdir.name)
    
    if available:
        click.echo("Available benchmark results:")
        for name in sorted(available):
            click.echo(f"  • results/reports/{name}/")
        click.echo("\nUsage: python main.py report -r <path>")
        click.echo(f"Example: python main.py report -r results/reports/{available[0]}")
    else:
        click.echo("No benchmark results found.", err=True)
        click.echo("Run a benchmark first: python main.py run -c <config>", err=True)
