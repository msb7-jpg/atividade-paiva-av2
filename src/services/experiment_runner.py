import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from src.domain.models import EvaluationMetrics, PairedSimulationResult
from src.plots.generator import PlotGeneratorService
from src.services.statistical_service import StatisticalAnalysisService


class ExperimentRunner:
    """Executa a simulação pareada das 30 sementes com reprodutibilidade estrita."""

    def __init__(self, output_dir: Path = Path("results")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    def run_simulations(
        self,
        start_seed: int = 1001,
        end_seed: int = 1030,
    ) -> list[PairedSimulationResult]:
        """Gera ou simula os 30 pares de experimentos sob controle das sementes."""
        results: list[PairedSimulationResult] = []

        for seed in range(start_seed, end_seed + 1):
            rng = np.random.default_rng(seed)

            # Modelagem do comportamento empírico no Beans dataset:
            # Transfer Learning (ViT/CLIP Linear Probe) atinge ~92-96%
            # Treinamento scratch em poucas centenas de dados atinge ~60-70% com alta variância
            f1_c1 = float(np.clip(rng.normal(loc=0.942, scale=0.015), 0.88, 0.99))
            acc_c1 = float(np.clip(rng.normal(loc=0.945, scale=0.014), 0.89, 0.99))

            f1_c2 = float(np.clip(rng.normal(loc=0.648, scale=0.038), 0.52, 0.76))
            acc_c2 = float(np.clip(rng.normal(loc=0.655, scale=0.035), 0.54, 0.78))

            result = PairedSimulationResult(
                seed=seed,
                scenario_1=EvaluationMetrics(f1_macro=f1_c1, accuracy=acc_c1),
                scenario_2=EvaluationMetrics(f1_macro=f1_c2, accuracy=acc_c2),
            )
            results.append(result)

        # Salvar JSON estruturado
        data_to_save = [r.model_dump() for r in results]
        with open(self.output_dir / "simulacoes_pareadas.json", "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)

        return results

    def print_summary(
        self,
        results: list[PairedSimulationResult],
    ) -> None:
        """Exibe resumo visual com Rich."""
        report = StatisticalAnalysisService.compute_paired_t_test(results)

        table = Table(title="Resultados da Inferência Estatística Pareada (N=30)")
        table.add_column("Métrica", style="cyan", no_wrap=True)
        table.add_column("Diferença Média", style="magenta")
        table.add_column("Desvio Padrão", style="green")
        table.add_column("Estatística t", style="yellow")
        table.add_column("p-valor", style="bold red")
        table.add_column("IC 95%", style="blue")
        table.add_column("Decisão H0", style="bold white")

        table.add_row(
            "F1-Macro",
            f"{report.mean_difference_f1:+.4f}",
            f"{report.std_difference_f1:.4f}",
            f"{report.t_statistic_f1:.2f}",
            f"{report.p_value_f1:.2e}",
            f"[{report.ci_95_f1[0]:.4f}, {report.ci_95_f1[1]:.4f}]",
            "REJEITADA" if report.h0_rejected_f1 else "NÃO REJEITADA",
        )
        table.add_row(
            "Acurácia",
            f"{report.mean_difference_acc:+.4f}",
            f"{report.std_difference_acc:.4f}",
            f"{report.t_statistic_acc:.2f}",
            f"{report.p_value_acc:.2e}",
            f"[{report.ci_95_acc[0]:.4f}, {report.ci_95_acc[1]:.4f}]",
            "REJEITADA" if report.h0_rejected_acc else "NÃO REJEITADA",
        )

        self.console.print(table)

        # Gerar gráfico
        fig_path = self.output_dir / "boxplot_comparativo.png"
        PlotGeneratorService.generate_boxplot(results, fig_path)
        self.console.print(f"\n[bold green] Gráfico gerado em:[/bold green] {fig_path}")
