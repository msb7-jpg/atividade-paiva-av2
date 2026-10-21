from pathlib import Path

import matplotlib.pyplot as plt


def generate_methodology_diagram(output_path: Path) -> Path:
    """Gera um fluxograma vertical em proporção de coluna IEEE com alta legibilidade."""
    # Proporção ideal para 1 coluna IEEE: 3.5 polegadas de largura por 5.2 de altura
    fig, ax = plt.subplots(figsize=(3.5, 5.2), dpi=300)
    ax.axis("off")

    # Configuração tipográfica acadêmica estilo IEEE (Serif)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "DejaVu Serif", "Times New Roman"],
        }
    )

    # Estilos de caixas oficiais e sóbrios
    style_neutral = {
        "boxstyle": "round,pad=0.45",
        "facecolor": "#f8f9fa",
        "edgecolor": "#2c3e50",
        "linewidth": 1.2,
    }
    style_c1 = {
        "boxstyle": "round,pad=0.45",
        "facecolor": "#ebf5fb",
        "edgecolor": "#1b4f72",
        "linewidth": 1.4,
    }
    style_c2 = {
        "boxstyle": "round,pad=0.45",
        "facecolor": "#fef9e7",
        "edgecolor": "#7d6608",
        "linewidth": 1.4,
    }
    style_eval = {
        "boxstyle": "round,pad=0.45",
        "facecolor": "#eafaf1",
        "edgecolor": "#145a32",
        "linewidth": 1.4,
    }

    # 1. Entrada: Dataset
    ax.text(
        0.5,
        0.92,
        "1. Conjunto de Dados: Chest X-Ray (2 Classes)\n"
        "• 4.077 Treino  ·  582 Teste Independente\n"
        "• Radiografias de Tórax: Normal vs. Pneumonia",
        ha="center",
        va="center",
        bbox=style_neutral,
        fontsize=8.5,
        linespacing=1.3,
    )

    # Seta 1 -> 2
    ax.annotate(
        "",
        xy=(0.5, 0.79),
        xytext=(0.5, 0.84),
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#2c3e50", "mutation_scale": 12},
    )

    # 2. Técnica A
    ax.text(
        0.5,
        0.72,
        "Técnica A (Fixa / Pré-processamento):\n"
        "Data Augmentation Estocástico\n"
        "• Resize (224×224 px) · Flip Horizontal (p=0.5)\n"
        "• Rotação Estocástica (±10°) · Jitter Fotométrico\n"
        "• Semente s_i fixa por simulação pareada",
        ha="center",
        va="center",
        bbox=style_neutral,
        fontsize=8.5,
        linespacing=1.3,
    )

    # Seta ramificação 2 -> C1 e C2
    ax.annotate(
        "",
        xy=(0.28, 0.54),
        xytext=(0.42, 0.62),
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#1b4f72", "mutation_scale": 11},
    )
    ax.annotate(
        "",
        xy=(0.72, 0.54),
        xytext=(0.58, 0.62),
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#7d6608", "mutation_scale": 11},
    )

    # 3. Variável Independente: C1 vs C2
    ax.text(
        0.26,
        0.47,
        "Cenário 1 (Proposto)\n"
        "Transfer Learning\n"
        "ResNet-50 ImageNet-1k\n"
        "Otimizador Adam\n"
        "(lr = 1e-4, L2 = 1e-4)",
        ha="center",
        va="center",
        bbox=style_c1,
        fontsize=8.0,
        fontweight="bold",
        color="#1b4f72",
        linespacing=1.2,
    )

    ax.text(
        0.74,
        0.47,
        "Cenário 2 (Contrafactual)\n"
        "Treinamento do Zero\n"
        "He Normal (Kaiming)\n"
        "Linear Warmup +\n"
        "Cosine Annealing",
        ha="center",
        va="center",
        bbox=style_c2,
        fontsize=8.0,
        fontweight="bold",
        color="#7d6608",
        linespacing=1.2,
    )

    # Setas convergência C1/C2 -> Técnica B
    ax.annotate(
        "",
        xy=(0.42, 0.35),
        xytext=(0.28, 0.39),
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#1b4f72", "mutation_scale": 11},
    )
    ax.annotate(
        "",
        xy=(0.58, 0.35),
        xytext=(0.72, 0.39),
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#7d6608", "mutation_scale": 11},
    )

    # 4. Técnica B (Explicabilidade e Auditoria Multimodal)
    ax.text(
        0.5,
        0.28,
        "Técnica B (Fixa / Explicabilidade & Auditoria):\n"
        "Grad-CAM + LMM Local (Gemma 4)\n"
        "• Grad-CAM na última camada convolucional (layer4)\n"
        "• Extração de predição e confiança reais na GPU\n"
        "• Auditoria de laudo clínico via LMM em localhost:8080",
        ha="center",
        va="center",
        bbox=style_neutral,
        fontsize=8.3,
        linespacing=1.3,
    )

    # Seta 4 -> Avaliação
    ax.annotate(
        "",
        xy=(0.5, 0.165),
        xytext=(0.5, 0.205),
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#145a32", "mutation_scale": 12},
    )

    # 5. Avaliação e Teste Pareado
    ax.text(
        0.5,
        0.09,
        "Protocolo Estatístico Pareado (N = 30):\n"
        "• 30 Sementes Idênticas Pareadas (1001 a 1030)\n"
        "• Avaliação no mesmo conjunto de teste (582 imagens)\n"
        "• Teste t de Student pareado bilateral (α = 0,05)",
        ha="center",
        va="center",
        bbox=style_eval,
        fontsize=8.5,
        fontweight="bold",
        color="#145a32",
        linespacing=1.3,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_methodology_diagram(Path("results/fluxograma_metodologia.png"))
    print("Fluxograma metodológico atualizado em results/fluxograma_metodologia.png")
