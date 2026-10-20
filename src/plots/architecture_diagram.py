from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def generate_methodology_diagram(output_path: Path) -> Path:
    """Gera um fluxograma vetorial claro da metodologia científica em blocos."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=300)
    ax.axis("off")

    # Configuração de estilo dos blocos
    box_blue = {
        "boxstyle": "round,pad=0.5",
        "facecolor": "#e8f1fa",
        "edgecolor": "#1f77b4",
        "linewidth": 1.5,
    }
    box_orange = {
        "boxstyle": "round,pad=0.5",
        "facecolor": "#fef0e7",
        "edgecolor": "#ff7f0e",
        "linewidth": 1.5,
    }
    box_gray = {
        "boxstyle": "round,pad=0.5",
        "facecolor": "#f2f2f2",
        "edgecolor": "#555555",
        "linewidth": 1.5,
    }
    box_green = {
        "boxstyle": "round,pad=0.5",
        "facecolor": "#eafaf1",
        "edgecolor": "#2ca02c",
        "linewidth": 1.5,
    }

    # 1. Entrada de Dados
    ax.text(
        0.08,
        0.5,
        "Beans Dataset\n(3 Classes)\nTreino: 1034\nValidação: 133\nTeste: 128",
        ha="center",
        va="center",
        bbox=box_gray,
        fontsize=7.5,
        fontweight="bold",
    )

    # 2. Técnica A
    ax.text(
        0.31,
        0.5,
        "Técnica A (Fixa)\nRedimensionamento 224x224\nNormalização RGB\nViT-B/32 Patch Encoder",
        ha="center",
        va="center",
        bbox=box_blue,
        fontsize=7.5,
        fontweight="bold",
    )

    # Setas de divisão
    ax.annotate(
        "",
        xy=(0.20, 0.5),
        xytext=(0.16, 0.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
    )
    ax.annotate(
        "",
        xy=(0.43, 0.73),
        xytext=(0.38, 0.55),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f77b4"),
    )
    ax.annotate(
        "",
        xy=(0.43, 0.27),
        xytext=(0.38, 0.45),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#ff7f0e"),
    )

    # 3. Cenários C vs C_alt
    ax.text(
        0.56,
        0.75,
        "Cenário 1: Transfer Learning\nPesos Pré-Treinados CLIP\n(Features Semânticas Ricas)",
        ha="center",
        va="center",
        bbox=box_blue,
        fontsize=7.5,
    )
    ax.text(
        0.56,
        0.25,
        "Cenário 2: Aleatório (Scratch)\nPesos Aleatórios Iniciais\n(Sem Pré-Treino)",
        ha="center",
        va="center",
        bbox=box_orange,
        fontsize=7.5,
    )

    # Setas convergência
    ax.annotate(
        "",
        xy=(0.69, 0.55),
        xytext=(0.65, 0.70),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f77b4"),
    )
    ax.annotate(
        "",
        xy=(0.69, 0.45),
        xytext=(0.65, 0.30),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#ff7f0e"),
    )

    # 4. Técnica B (Cabeça Comum)
    ax.text(
        0.79,
        0.5,
        "Técnica B (Fixa)\nDropout (0.2)\nRegularização L2 (1e-4)\nLinear Probe + Softmax\nAdam (lr=1e-4)",
        ha="center",
        va="center",
        bbox=box_gray,
        fontsize=7.5,
        fontweight="bold",
    )

    # Seta para avaliação
    ax.annotate(
        "",
        xy=(0.89, 0.5),
        xytext=(0.86, 0.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
    )

    # 5. Avaliação Estatística Pareada
    ax.text(
        0.96,
        0.5,
        "Avaliação\n30 Sementes\n(1001-1030)\nTeste t Pareado\nIC 95%",
        ha="center",
        va="center",
        bbox=box_green,
        fontsize=7.5,
        fontweight="bold",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_methodology_diagram(Path("results/fluxograma_metodologia.png"))
    print("Fluxograma gerado em results/fluxograma_metodologia.png")
