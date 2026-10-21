import matplotlib.pyplot as plt

from src.repositories.dataset_repository import DatasetRepository


def plot_medical_samples(output_path: str = "results/amostras_clinicas_xray.png") -> None:
    """Gera o painel clínico de comparação: Radiografia Normal vs Pneumonia."""
    repo = DatasetRepository()
    samples = repo.get_fixed_audit_samples()

    normal_img, _ = samples["NORMAL"]
    pneumonia_img, _ = samples["PNEUMONIA"]

    fig, axes = plt.subplots(2, 1, figsize=(3.8, 6.8), dpi=300)

    axes[0].imshow(normal_img.resize((224, 224)), cmap="gray")
    axes[0].set_title("(A) Tórax Normal (Controle Sadio)", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    axes[1].imshow(pneumonia_img.resize((224, 224)), cmap="gray")
    axes[1].set_title(
        "(B) Pneumonia (Consolidação Alveolar)", fontsize=11, fontweight="bold", pad=8
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Painel clínico salvo em {output_path}")


if __name__ == "__main__":
    plot_medical_samples()
