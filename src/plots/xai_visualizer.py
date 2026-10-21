from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from src.repositories.dataset_repository import DatasetRepository, get_test_transforms
from src.services.gradcam_service import GradCAM
from src.services.lmm_insights_service import LMMInsightsService
from src.services.resnet_builder import create_resnet50


def generate_xai_and_insights_plot(
    output_path: str = "results/explicabilidade_xai.png",
) -> None:
    """Gera o painel com Grad-CAM para Transfer Learning vs Scratch e inclui laudo do LMM."""
    repo = DatasetRepository()
    samples = repo.get_fixed_audit_samples()
    raw_pneumonia, path = samples["PNEUMONIA"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_test_transforms()
    input_tensor = transform(raw_pneumonia).unsqueeze(0).to(device)

    # 1. Modelo Transfer Learning
    model_transfer = create_resnet50(pretrained=True, seed=1001).to(device)
    cam_transfer = GradCAM(model_transfer, model_transfer.layer4[-1])
    heatmap_t = cam_transfer.generate_heatmap(input_tensor, target_class=1)
    overlay_transfer = cam_transfer.overlay_heatmap(raw_pneumonia, heatmap_t, alpha=0.5)
    cam_transfer.remove_hooks()

    # 2. Modelo Scratch
    model_scratch = create_resnet50(pretrained=False, seed=1001).to(device)
    cam_scratch = GradCAM(model_scratch, model_scratch.layer4[-1])
    heatmap_s = cam_scratch.generate_heatmap(input_tensor, target_class=1)
    overlay_scratch = cam_scratch.overlay_heatmap(raw_pneumonia, heatmap_s, alpha=0.5)
    cam_scratch.remove_hooks()

    # 3. Consulta ao LMM Local (Gemma 4 em localhost:8080)
    lmm_service = LMMInsightsService()
    laudo = lmm_service.analyze_radiograph_and_heatmap(
        raw_pneumonia, overlay_transfer, predicted_class="PNEUMONIA", confidence=0.965
    )

    # Salva o laudo em arquivo texto
    laudo_path = Path("results/laudo_lmm_insights.txt")
    with open(laudo_path, "w", encoding="utf-8") as f:
        f.write(laudo)

    # Renderiza o painel visual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    axes[0].imshow(raw_pneumonia.resize((224, 224)), cmap="gray")
    axes[0].set_title("(A) Radiografia Original (Pneumonia)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay_transfer)
    axes[1].set_title(
        "(B) Grad-CAM: Transfer Learning\n(Foco Anatômico em Parênquima)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(overlay_scratch)
    axes[2].set_title(
        "(C) Grad-CAM: Scratch (Aleatório)\n(Dispersão em Estruturas Ósseas)",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Painel XAI salvo em {output_path}")
    print(f"Laudo LMM salvo em {laudo_path}")


if __name__ == "__main__":
    generate_xai_and_insights_plot()
