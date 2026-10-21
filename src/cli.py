from pathlib import Path
from rich.console import Console
from rich.table import Table
import typer

from src.plots.generator import PlotGeneratorService
from src.plots.medical_samples import plot_medical_samples
from src.plots.xai_visualizer import generate_xai_and_insights_plot
from src.services.experiment_runner import ExperimentRunner
from src.services.llama_lifecycle import LlamaLifecycleManager
from src.services.statistical_service import StatisticalAnalysisService

app = typer.Typer(
    help="CLI para execução empírica real e inferência estatística da 2ª Avaliação (CESAR School)"
)
console = Console()


@app.command()
def run(
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Diretório de saída"),
    start_seed: int = typer.Option(1001, "--start-seed", help="Semente inicial"),
    end_seed: int = typer.Option(1030, "--end-seed", help="Semente final"),
    epochs: int = typer.Option(2, "--epochs", "-e", help="Épocas por simulação"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Tamanho do batch"),
    skip_lmm: bool = typer.Option(False, "--skip-lmm", help="Ignorar laudo multimodal LMM"),
) -> None:
    """Executa as 30 simulações pareadas na GPU, gera gráficos, laudo do LMM e relatório formal."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seeds = list(range(start_seed, end_seed + 1))

    llama_mgr = LlamaLifecycleManager()

    # 1. Gerenciar VRAM: desliga temporariamente o llama-server se estiver ocupando memória
    console.print("\n[bold cyan]1. Gerenciamento de Compute e VRAM:[/bold cyan]")
    if llama_mgr.is_running():
        console.print("   [yellow]Liberando VRAM da GPU para o treinamento...[/yellow]")
        llama_mgr.stop_server()
        console.print("   [green]VRAM 100% liberada na GPU RTX 5060.[/green]")
    else:
        console.print("   [green]VRAM já disponível para treinamento na GPU.[/green]")

    # 2. Executar 30 simulações pareadas reais
    console.print(f"\n[bold cyan]2. Executando {len(seeds)} Simulações Pareadas (60 Treinamentos):[/bold cyan]")
    runner = ExperimentRunner(seeds=seeds, epochs_per_simulation=epochs, batch_size=batch_size)
    results = runner.run_all_simulations()
    runner.save_results(results, filepath=out_path / "simulacoes_pareadas.json")

    # 3. Análise estatística
    console.print("\n[bold cyan]3. Inferência Estatística Formal (Teste t Pareado Bilateral):[/bold cyan]")
    report = StatisticalAnalysisService.compute_paired_t_test(results)

    table = Table(title=f"Resultados do Teste t Pareado Bilateral (N={len(results)}, gl={len(results)-1})")
    table.add_column("Métrica", style="cyan", no_wrap=True)
    table.add_column("Dif. Média (d̄)", style="magenta")
    table.add_column("Desvio (sd)", style="blue")
    table.add_column("Estatística t", style="green")
    table.add_column("Valor-p", style="bold yellow")
    table.add_column("IC 95%", style="white")
    table.add_column("Decisão (α=0.05)", style="bold red")

    table.add_row(
        "F1-Macro",
        f"{report.mean_difference_f1:+.4f}",
        f"{report.std_difference_f1:.4f}",
        f"{report.t_statistic_f1:.2f}",
        f"{report.p_value_f1:.2e}",
        f"[{report.ci_95_f1[0]:.4f}, {report.ci_95_f1[1]:.4f}]",
        "Rejeita H0" if report.h0_rejected_f1 else "Aceita H0",
    )
    table.add_row(
        "Acurácia",
        f"{report.mean_difference_acc:+.4f}",
        f"{report.std_difference_acc:.4f}",
        f"{report.t_statistic_acc:.2f}",
        f"{report.p_value_acc:.2e}",
        f"[{report.ci_95_acc[0]:.4f}, {report.ci_95_acc[1]:.4f}]",
        "Rejeita H0" if report.h0_rejected_acc else "Aceita H0",
    )
    console.print(table)

    # 4. Geração de figuras
    console.print("\n[bold cyan]4. Gerando Gráficos e Painéis Clínicos em Alta Resolução:[/bold cyan]")
    PlotGeneratorService.generate_boxplot(results, output_path=out_path / "boxplot_comparativo.png")
    console.print("   [green]✓ Boxplot comparativo salvo em results/boxplot_comparativo.png[/green]")
    plot_medical_samples(output_path=str(out_path / "amostras_clinicas_xray.png"))
    console.print("   [green]✓ Painel clínico salvo em results/amostras_clinicas_xray.png[/green]")

    # 5. Grad-CAM + LMM Insights
    if not skip_lmm:
        console.print("\n[bold cyan]5. Ativando LMM Localhost para Explicabilidade e Laudo Clínico:[/bold cyan]")
        if not llama_mgr.is_running():
            console.print("   [yellow]Iniciando llama-server sob demanda na porta 8080...[/yellow]")
            llama_mgr.start_server(timeout=30)

        generate_xai_and_insights_plot(output_path=str(out_path / "explicabilidade_xai.png"))
        console.print("   [green]✓ Painel XAI Grad-CAM e laudo do LMM salvos com sucesso.[/green]")

    console.print("\n[bold green]Pipeline concluída com sucesso![/bold green]\n")


if __name__ == "__main__":
    app()
