from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.domain.models import PairedSimulationResult


class PlotGeneratorService:
    """Gera figuras científicas prontas para publicação no padrão IEEE de 2 colunas."""

    @staticmethod
    def generate_boxplot(
        results: list[PairedSimulationResult],
        output_path: Path,
    ) -> Path:
        """Gera boxplot comparativo de F1-macro e Acurácia entre os dois cenários."""
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({"font.size": 9})

        data_rows = []
        for r in results:
            data_rows.append(
                {
                    "Cenário": "Transfer Learning",
                    "F1-Macro": r.scenario_1.f1_macro,
                    "Acurácia": r.scenario_1.accuracy,
                }
            )
            data_rows.append(
                {
                    "Cenário": "Scratch (Warmup)",
                    "F1-Macro": r.scenario_2.f1_macro,
                    "Acurácia": r.scenario_2.accuracy,
                }
            )

        df = pd.DataFrame(data_rows)

        fig, axes = plt.subplots(2, 1, figsize=(3.8, 5.4), sharex=True)

        # Plot F1
        sns.boxplot(
            ax=axes[0],
            data=df,
            x="Cenário",
            y="F1-Macro",
            hue="Cenário",
            palette=["#1f77b4", "#ff7f0e"],
            legend=False,
            width=0.4,
            fliersize=3,
        )
        axes[0].set_title("(A) Distribuição F1-Macro (N=30)", fontweight="bold", fontsize=10, pad=6)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("F1-Macro", fontweight="bold", fontsize=9)

        # Plot Acurácia
        sns.boxplot(
            ax=axes[1],
            data=df,
            x="Cenário",
            y="Acurácia",
            hue="Cenário",
            palette=["#1f77b4", "#ff7f0e"],
            legend=False,
            width=0.4,
            fliersize=3,
        )
        axes[1].set_title("(B) Distribuição Acurácia (N=30)", fontweight="bold", fontsize=10, pad=6)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Acurácia", fontweight="bold", fontsize=9)
        axes[1].tick_params(axis="x", labelsize=9)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path
