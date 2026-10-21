import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from src.repositories.dataset_repository import DatasetRepository, IDX_TO_CLASS, get_test_transforms
from src.services.gradcam_service import GradCAM
from src.services.lmm_insights_service import LMMInsightsService, laudo_or_unavailable_notice
from src.services.resnet_builder import create_resnet50

logger = logging.getLogger(__name__)


def load_or_train_model(
    pretrained: bool,
    seed: int,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Carrega os pesos treinados do checkpoint ou executa um treino pontual e salva."""
    model = create_resnet50(pretrained=pretrained, seed=seed).to(device)

    if checkpoint_path.exists():
        logger.info(f"Carregando pesos reais de {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            f"Checkpoint {checkpoint_path} não encontrado! Executando treinamento rápido (1 época) para obter pesos reais..."
        )
        from src.services.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(seeds=[seed], epochs_per_simulation=1, device=str(device))
        train_loader = runner.repo.get_train_loader(batch_size=64, seed=seed, shuffle=True)
        test_loader = runner.repo.get_test_loader(batch_size=64)
        _, model = runner.train_and_evaluate(
            pretrained=pretrained,
            seed=seed,
            train_loader=train_loader,
            test_loader=test_loader,
            checkpoint_path=checkpoint_path,
        )

    model.eval()
    return model


def generate_xai_and_insights_plot(
    output_path: str = "results/explicabilidade_xai.png",
    laudo_output_path: str = "results/laudo_lmm_insights.txt",
    seed: int = 1001,
    checkpoints_dir: str | Path = "results/checkpoints",
) -> dict:
    """Gera o painel com Grad-CAM para Transfer Learning vs Scratch com pesos treinados reais e laudo do LMM."""
    repo = DatasetRepository()
    samples = repo.get_fixed_audit_samples()
    raw_pneumonia, path = samples["PNEUMONIA"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_test_transforms()
    input_tensor = transform(raw_pneumonia).unsqueeze(0).to(device)

    cp_dir = Path(checkpoints_dir)
    cp_transfer = cp_dir / f"seed_{seed}_transfer.pt"
    cp_scratch = cp_dir / f"seed_{seed}_scratch.pt"

    # 1. Modelo Transfer Learning (Carregando Pesos Treinados Reais)
    model_transfer = load_or_train_model(
        pretrained=True, seed=seed, checkpoint_path=cp_transfer, device=device
    )

    # Inferência Real (Transfer Learning)
    with torch.no_grad():
        logits_t = model_transfer(input_tensor)
        probs_t = torch.softmax(logits_t, dim=1)
        pred_idx_t = int(torch.argmax(probs_t, dim=1).item())
        conf_t = float(probs_t[0, pred_idx_t].item())
        predicted_class_t = IDX_TO_CLASS[pred_idx_t]

    cam_transfer = GradCAM(model_transfer, model_transfer.layer4[-1])
    heatmap_t = cam_transfer.generate_heatmap(input_tensor, target_class=pred_idx_t)
    overlay_transfer = cam_transfer.overlay_heatmap(raw_pneumonia, heatmap_t, alpha=0.5)
    cam_transfer.remove_hooks()

    # 2. Modelo Scratch (Carregando Pesos Treinados Reais)
    model_scratch = load_or_train_model(
        pretrained=False, seed=seed, checkpoint_path=cp_scratch, device=device
    )

    # Inferência Real (Scratch)
    with torch.no_grad():
        logits_s = model_scratch(input_tensor)
        probs_s = torch.softmax(logits_s, dim=1)
        pred_idx_s = int(torch.argmax(probs_s, dim=1).item())
        conf_s = float(probs_s[0, pred_idx_s].item())
        predicted_class_s = IDX_TO_CLASS[pred_idx_s]

    cam_scratch = GradCAM(model_scratch, model_scratch.layer4[-1])
    heatmap_s = cam_scratch.generate_heatmap(input_tensor, target_class=pred_idx_s)
    overlay_scratch = cam_scratch.overlay_heatmap(raw_pneumonia, heatmap_s, alpha=0.5)
    cam_scratch.remove_hooks()

    # 3. Consulta ao LMM Local (com os dados reais inferidos pelo modelo)
    lmm_service = LMMInsightsService()
    laudo = laudo_or_unavailable_notice(
        lmm_service,
        raw_pneumonia,
        overlay_transfer,
        predicted_class=predicted_class_t,
        confidence=conf_t,
    )

    # Salva o laudo em arquivo texto
    laudo_path = Path(laudo_output_path)
    laudo_path.parent.mkdir(parents=True, exist_ok=True)
    with open(laudo_path, "w", encoding="utf-8") as f:
        f.write(laudo)

    # 4. Renderiza o painel visual com anotações empíricas reais (layout vertical para coluna IEEE)
    fig, axes = plt.subplots(3, 1, figsize=(3.8, 9.8), dpi=300)

    axes[0].imshow(raw_pneumonia.resize((224, 224)), cmap="gray")
    axes[0].set_title(
        "(A) Radiografia Original (Pneumonia)",
        fontsize=10,
        fontweight="bold",
        pad=6,
    )
    axes[0].axis("off")

    axes[1].imshow(overlay_transfer)
    axes[1].set_title(
        f"(B) Grad-CAM: Transfer Learning\nPredição: {predicted_class_t} ({conf_t * 100:.1f}%)",
        fontsize=10,
        fontweight="bold",
        color="#0d47a1",
        pad=6,
    )
    axes[1].axis("off")

    axes[2].imshow(overlay_scratch)
    axes[2].set_title(
        f"(C) Grad-CAM: Scratch (Warmup)\nPredição: {predicted_class_s} ({conf_s * 100:.1f}%)",
        fontsize=10,
        fontweight="bold",
        color="#b71c1c",
        pad=6,
    )
    axes[2].axis("off")

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

    logger.info(f"Painel XAI salvo em {out_file}")
    logger.info(f"Laudo LMM salvo em {laudo_path}")

    return {
        "transfer": {
            "predicted_class": predicted_class_t,
            "confidence": conf_t,
        },
        "scratch": {
            "predicted_class": predicted_class_s,
            "confidence": conf_s,
        },
        "laudo": laudo,
    }


if __name__ == "__main__":
    generate_xai_and_insights_plot()
