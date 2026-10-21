from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from src.domain.models import EvaluationMetrics
from src.plots.architecture_diagram import generate_methodology_diagram
from src.plots.gallery_generator import generate_clinical_gallery
from src.plots.generator import PlotGeneratorService
from src.plots.medical_samples import plot_medical_samples
from src.plots.xai_visualizer import generate_xai_and_insights_plot
from src.services.experiment_runner import ExperimentRunner
from src.services.llama_lifecycle import LlamaLifecycleManager
from src.services.lmm_insights_service import LAUDO_NAO_GERADO
from src.services.rocm_compat import apply_rocm_windows_workarounds
from src.services.statistical_service import StatisticalAnalysisService

apply_rocm_windows_workarounds()

app = typer.Typer(
    help="CLI para execução empírica real, persistência e inferência estatística da 2ª Avaliação (CESAR School)"
)
console = Console()


def _format_confusion_matrix(header: str, m: EvaluationMetrics) -> str:
    """Formata a matriz de confusão de um cenário a partir das métricas salvas em disco."""
    tp, tn, fp, fn = m.tp, m.tn, m.fp, m.fn
    assert tp is not None and tn is not None and fp is not None and fn is not None
    positives = tp + fn
    negatives = tn + fp
    total = positives + negatives
    correct = tp + tn
    errors = fp + fn
    return (
        f"{header}\n"
        f"  • Acertos Totais: [bold]{correct} / {total} ({correct / total:.2%})[/bold]  |  "
        f"Erros: [bold]{errors} / {total} ({errors / total:.2%})[/bold]\n"
        f"  • Sensibilidade (Pneumonia): [bold]{tp / positives:.2%}[/bold] "
        f"({tp} de {positives} detectadas, {fn} Falsos Negativos)\n"
        f"  • Especificidade (Sadio):   [bold]{tn / negatives:.2%}[/bold] "
        f"({tn} de {negatives} confirmadas sadias, {fp} Falsos Positivos)"
    )


@app.command()
def status(
    results_dir: str = typer.Option("results", "--results-dir", "-d", help="Diretório de resultados"),
) -> None:
    """Inspeciona o progresso dos experimentos, sementes executadas e checkpoints salvos."""
    base_dir = Path(results_dir)
    runs_dir = base_dir / "runs"
    cp_dir = base_dir / "checkpoints"
    consolidated_path = base_dir / "simulacoes_pareadas.json"

    saved_runs = ExperimentRunner.get_all_saved_runs(results_dir=base_dir)
    all_seeds = list(range(1001, 1031))
    completed_seeds = {r.seed: r for r in saved_runs}

    console.print("\n[bold cyan]=== Painel de Status dos Experimentos (N=30) ===[/bold cyan]\n")

    summary_text = (
        f"[bold]Total Planejado:[/bold] {len(all_seeds)} sementes (1001 a 1030)\n"
        f"[bold]Executadas e Persistidas:[/bold] {len(saved_runs)}/{len(all_seeds)} "
        f"({'[green]100% Concluído[/green]' if len(saved_runs) >= len(all_seeds) else f'[yellow]{len(saved_runs)/len(all_seeds)*100:.1f}% Concluído[/yellow]'})\n"
        f"[bold]Arquivo Consolidado:[/bold] {'[green]Presente[/green]' if consolidated_path.exists() else '[red]Ausente[/red]'}\n"
        f"[bold]Diretório de Runs:[/bold] {runs_dir}\n"
        f"[bold]Diretório de Checkpoints:[/bold] {cp_dir}"
    )
    console.print(Panel(summary_text, title="Inventário de Execução", expand=False))

    # Tabela de sementes e checkpoints
    table = Table(title="Detalhamento das Rodadas e Modelos Treinados em Disco")
    table.add_column("Semente", style="cyan", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("F1 Cenário 1 (Transfer)", style="green", justify="right")
    table.add_column("F1 Cenário 2 (Scratch)", style="yellow", justify="right")
    table.add_column("Diferença F1", style="magenta", justify="right")
    table.add_column("Checkpoints Salvos (.pt)", style="blue", justify="center")

    for s in all_seeds:
        has_cp_t = (cp_dir / f"seed_{s}_transfer.pt").exists()
        has_cp_s = (cp_dir / f"seed_{s}_scratch.pt").exists()
        cp_label = []
        if has_cp_t:
            cp_label.append("Transfer")
        if has_cp_s:
            cp_label.append("Scratch")
        cp_str = " + ".join(cp_label) if cp_label else "[dim]-[/dim]"

        if s in completed_seeds:
            run_data = completed_seeds[s]
            diff = run_data.f1_difference
            table.add_row(
                str(s),
                "[green]✓ Salvo[/green]",
                f"{run_data.scenario_1.f1_macro:.4f}",
                f"{run_data.scenario_2.f1_macro:.4f}",
                f"{diff:+.4f}",
                cp_str,
            )
        else:
            table.add_row(
                str(s),
                "[dim red]Pendente[/dim red]",
                "-",
                "-",
                "-",
                cp_str,
            )

    console.print(table)

    ref_seed = 1001
    if ref_seed in completed_seeds:
        ref_run = completed_seeds[ref_seed]
        if ref_run.scenario_1.has_confusion_matrix and ref_run.scenario_2.has_confusion_matrix:
            cm_panel = (
                _format_confusion_matrix(
                    "[bold green]Cenário 1 (Transfer Learning - ResNet-50 ImageNet):[/bold green]",
                    ref_run.scenario_1,
                )
                + "\n\n"
                + _format_confusion_matrix(
                    "[bold yellow]Cenário 2 (Scratch - Treinamento do Zero + Cosine):[/bold yellow]",
                    ref_run.scenario_2,
                )
            )
        else:
            cm_panel = (
                "[yellow]Esta run foi salva sem a matriz de confusão (apenas F1 e acurácia).[/yellow]\n"
                f"Reexecute com: python -m src.cli run --start-seed {ref_seed} --end-seed {ref_seed} --no-resume"
            )
        console.print(
            Panel(cm_panel, title=f"Auditoria Clínica Amostral (Semente de Referência {ref_seed})", expand=False)
        )

    console.print(
        "\n[dim]Dica: Para executar sementes pendentes ou reusar runs salvas, use 'python -m src.cli run --resume'[/dim]\n"
    )


@app.command()
def run(
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Diretório de saída"),
    start_seed: int = typer.Option(1001, "--start-seed", help="Semente inicial"),
    end_seed: int = typer.Option(1030, "--end-seed", help="Semente final"),
    epochs: int = typer.Option(2, "--epochs", "-e", help="Épocas por simulação"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Tamanho do batch"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Reutilizar runs já concluídas em disco"),
    save_checkpoints: bool = typer.Option(
        False, "--save-checkpoints", help="Salvar checkpoints .pt de todas as sementes (além da referência 1001)"
    ),
    skip_lmm: bool = typer.Option(False, "--skip-lmm", help="Ignorar laudo multimodal LMM"),
) -> None:
    """Executa simulações pareadas na GPU com persistência granular, estatística e gráficos."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seeds = list(range(start_seed, end_seed + 1))

    llama_mgr = LlamaLifecycleManager()

    # 1. Gerenciar VRAM: desliga temporariamente o llama-server se estiver ocupando memória
    console.print("\n[bold cyan]1. Gerenciamento de Compute e VRAM:[/bold cyan]")
    if llama_mgr.is_running():
        console.print("   [yellow]Liberando VRAM da GPU para o treinamento...[/yellow]")
        llama_mgr.stop_server()
        console.print("   [green]VRAM liberada na GPU.[/green]")
    else:
        console.print("   [green]VRAM já disponível para treinamento na GPU.[/green]")

    # 2. Executar simulações com persistência granular
    console.print(
        f"\n[bold cyan]2. Executando {len(seeds)} Simulações Pareadas (resume={resume}):[/bold cyan]"
    )
    runner = ExperimentRunner(
        seeds=seeds,
        epochs_per_simulation=epochs,
        batch_size=batch_size,
        results_dir=out_path,
        checkpoint_seeds=seeds if save_checkpoints else [1001],
    )
    results = runner.run_all_simulations(resume=resume, save_all_checkpoints=save_checkpoints)

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

        res_xai = generate_xai_and_insights_plot(
            output_path=str(out_path / "explicabilidade_xai.png"),
            seed=1001,
            checkpoints_dir=out_path / "checkpoints",
        )
        console.print(
            f"   [green]✓ Painel XAI Grad-CAM gerado com predição real: "
            f"{res_xai['transfer']['predicted_class']} ({res_xai['transfer']['confidence']*100:.1f}%).[/green]"
        )

    console.print("\n[bold green]Pipeline concluída com sucesso![/bold green]\n")


@app.command()
def xai(
    seed: int = typer.Option(1001, "--seed", "-s", help="Semente do modelo treinado a utilizar"),
    output: str = typer.Option("results/explicabilidade_xai.png", "--output", "-o", help="Caminho do painel PNG"),
    laudo_output: str = typer.Option(
        "results/laudo_lmm_insights.txt", "--laudo-output", "-l", help="Caminho do arquivo de laudo TXT"
    ),
    checkpoints_dir: str = typer.Option("results/checkpoints", "--checkpoints-dir", "-c", help="Pasta de checkpoints"),
    skip_server: bool = typer.Option(False, "--skip-server", help="Não iniciar o llama-server caso offline"),
) -> None:
    """Executa auditoria XAI (Grad-CAM) e laudo LMM sobre um modelo treinado real."""
    console.print(f"\n[bold cyan]=== Auditoria XAI & Consulta LMM (Semente {seed}) ===[/bold cyan]\n")

    llama_mgr = LlamaLifecycleManager()
    if not skip_server and not llama_mgr.is_running():
        console.print("[yellow]Servidor LMM não detectado em localhost:8080. Tentando iniciar...[/yellow]")
        llama_mgr.start_server(timeout=30)

    console.print("[cyan]Realizando inferência empírica na GPU e gerando mapas Grad-CAM...[/cyan]")
    res = generate_xai_and_insights_plot(
        output_path=output,
        laudo_output_path=laudo_output,
        seed=seed,
        checkpoints_dir=checkpoints_dir,
    )

    console.print(f"\n[bold green]Inferência do Modelo Transfer Learning:[/bold green]")
    console.print(f"  • Classe Predita: [bold]{res['transfer']['predicted_class']}[/bold]")
    console.print(f"  • Confiança: [bold]{res['transfer']['confidence'] * 100:.2f}%[/bold]")

    console.print(f"\n[bold yellow]Inferência do Modelo Scratch (Warmup):[/bold yellow]")
    console.print(f"  • Classe Predita: [bold]{res['scratch']['predicted_class']}[/bold]")
    console.print(f"  • Confiança: [bold]{res['scratch']['confidence'] * 100:.2f}%[/bold]")

    console.print(f"\n[bold magenta]Parecer / Laudo Multimodal Gerado:[/bold magenta]")
    if res["laudo"].startswith(LAUDO_NAO_GERADO):
        console.print(Panel(res["laudo"], title="LMM indisponível", border_style="yellow", expand=False))
    else:
        console.print(Panel(res["laudo"], title="Laudo Clínico Estruturado (LMM)", expand=False))
    console.print(f"\n[green]✓ Arquivos salvos: {output} e {laudo_output}[/green]\n")


@app.command()
def gallery(
    output_dir: str = typer.Option("results/gallery", "--output-dir", "-o", help="Diretório da galeria"),
    seed: int = typer.Option(1001, "--seed", "-s", help="Semente do modelo treinado a utilizar"),
    samples_per_category: int = typer.Option(
        2, "--samples", "-n", help="Número de amostras por quadrante (TP, TN, FP, FN)"
    ),
    checkpoints_dir: str = typer.Option("results/checkpoints", "--checkpoints-dir", "-c", help="Pasta de checkpoints"),
) -> None:
    """Gera galeria clínica multi-casos (Verdadeiros Positivos, Normais e Erros) com laudos LMM."""
    console.print(f"\n[bold cyan]=== Gerando Galeria Clínica Multi-Casos (Semente {seed}) ===[/bold cyan]\n")
    out = generate_clinical_gallery(
        output_dir=output_dir,
        seed=seed,
        samples_per_category=samples_per_category,
        checkpoints_dir=checkpoints_dir,
    )
    console.print(f"\n[bold green]✓ Galeria completa com laudos gerada com sucesso em: {out}[/bold green]")
    console.print(f"[dim]Consulte o catálogo em: {Path(out) / 'README.md'}[/dim]\n")


@app.command()
def diagram(
    output: str = typer.Option(
        "results/fluxograma_metodologia.png", "--output", "-o", help="Caminho do diagrama PNG"
    ),
) -> None:
    """Regenera o fluxograma metodológico oficial IEEE."""
    path = Path(output)
    generate_methodology_diagram(path)
    console.print(f"[green]✓ Fluxograma metodológico gerado com sucesso em: {path}[/green]")


if __name__ == "__main__":
    app()
