from pathlib import Path

import typer

from src.services.experiment_runner import ExperimentRunner

app = typer.Typer(help="CLI para execução e análise estatística da 2ª Avaliação")


@app.command()
def run(
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Diretório de saída"),
    start_seed: int = typer.Option(1001, "--start-seed", help="Semente inicial"),
    end_seed: int = typer.Option(1030, "--end-seed", help="Semente final"),
) -> None:
    """Executa as 30 simulações pareadas e gera estatísticas e gráficos."""
    runner = ExperimentRunner(output_dir=Path(output_dir))
    results = runner.run_simulations(start_seed=start_seed, end_seed=end_seed)
    runner.print_summary(results)


if __name__ == "__main__":
    app()
