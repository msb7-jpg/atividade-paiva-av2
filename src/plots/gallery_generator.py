import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.repositories.dataset_repository import DatasetRepository, IDX_TO_CLASS, get_test_transforms
from src.services.gradcam_service import GradCAM
from src.services.lmm_insights_service import LMMInsightsService
from src.services.resnet_builder import create_resnet50

logger = logging.getLogger(__name__)


def generate_clinical_gallery(
    output_dir: str | Path = "results/gallery",
    seed: int = 1001,
    samples_per_category: int = 2,
    checkpoints_dir: str | Path = "results/checkpoints",
) -> Path:
    """Gera uma galeria clínica com múltiplos casos representativos e laudos do LMM.

    Cobre os 4 quadrantes clínicos:
      - TP: Verdadeiro Positivo (Pneumonia detectada com sucesso)
      - TN: Verdadeiro Negativo (Tórax sadio confirmado)
      - FP: Falso Positivo (Sadio que a IA confundiu com pneumonia - auditoria de erro)
      - FN: Falso Negativo (Pneumonia não detectada - auditoria de erro crítico)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cp_dir = Path(checkpoints_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_test_transforms()
    repo = DatasetRepository()
    test_ds = repo.get_test_dataset()

    # 1. Carrega os modelos treinados
    cp_transfer = cp_dir / f"seed_{seed}_transfer.pt"
    cp_scratch = cp_dir / f"seed_{seed}_scratch.pt"

    model_transfer = create_resnet50(pretrained=False, seed=seed).to(device)
    if cp_transfer.exists():
        model_transfer.load_state_dict(
            torch.load(cp_transfer, map_location=device, weights_only=True)
        )
    model_transfer.eval()

    model_scratch = create_resnet50(pretrained=False, seed=seed).to(device)
    if cp_scratch.exists():
        model_scratch.load_state_dict(
            torch.load(cp_scratch, map_location=device, weights_only=True)
        )
    model_scratch.eval()

    # 2. Avaliação de todas as amostras de teste para categorização clínica
    logger.info("Mapeando os 4 quadrantes clínicos no conjunto de teste de 582 imagens...")
    categorized: dict[str, list[dict]] = {"TP": [], "TN": [], "FP": [], "FN": []}

    with torch.no_grad():
        for i in range(len(test_ds)):
            raw_img, target, img_path = test_ds.get_raw_image(i)
            tensor = transform(raw_img).unsqueeze(0).to(device)

            logits_t = model_transfer(tensor)
            probs_t = torch.softmax(logits_t, dim=1)
            pred_t = int(torch.argmax(probs_t, dim=1).item())
            conf_t = float(probs_t[0, pred_t].item())

            logits_s = model_scratch(tensor)
            probs_s = torch.softmax(logits_s, dim=1)
            pred_s = int(torch.argmax(probs_s, dim=1).item())
            conf_s = float(probs_s[0, pred_s].item())

            item = {
                "index": i,
                "raw_img": raw_img,
                "img_path": img_path,
                "target": target,
                "target_name": IDX_TO_CLASS[target],
                "pred_transfer": pred_t,
                "pred_transfer_name": IDX_TO_CLASS[pred_t],
                "conf_transfer": conf_t,
                "pred_scratch": pred_s,
                "pred_scratch_name": IDX_TO_CLASS[pred_s],
                "conf_scratch": conf_s,
            }

            if target == 1 and pred_t == 1:
                categorized["TP"].append(item)
            elif target == 0 and pred_t == 0:
                categorized["TN"].append(item)
            elif target == 0 and pred_t == 1:
                categorized["FP"].append(item)
            elif target == 1 and pred_t == 0:
                categorized["FN"].append(item)

    # 3. Seleciona os casos representativos
    selected_cases: list[tuple[str, dict]] = []
    cat_descriptions = {
        "TP": "Verdadeiro Positivo (Pneumonia Confirmada)",
        "TN": "Verdadeiro Negativo (Tórax Sadio)",
        "FP": "Falso Positivo (Erro: Sadio confundido com Pneumonia)",
        "FN": "Falso Negativo (Erro Crítico: Pneumonia não detectada)",
    }

    for cat in ["TP", "TN", "FP", "FN"]:
        pool = categorized[cat]
        # Ordena por confiança decrescente para casos corretos, e por confiança de erro para erros
        sorted_pool = sorted(pool, key=lambda x: x["conf_transfer"], reverse=True)
        chosen = sorted_pool[:samples_per_category]
        for c in chosen:
            selected_cases.append((cat, c))

    lmm_service = LMMInsightsService()
    gallery_metadata: list[dict] = []

    # 4. Geração dos painéis de alta resolução e laudos
    for idx, (cat, case) in enumerate(selected_cases, start=1):
        raw_img = case["raw_img"]
        tensor = transform(raw_img).unsqueeze(0).to(device)
        pred_t_name = case["pred_transfer_name"]
        conf_t = case["conf_transfer"]
        pred_s_name = case["pred_scratch_name"]
        conf_s = case["conf_scratch"]
        real_name = case["target_name"]

        # Grad-CAM Transfer
        cam_t = GradCAM(model_transfer, model_transfer.layer4[-1])
        heatmap_t = cam_t.generate_heatmap(tensor, target_class=case["pred_transfer"])
        overlay_t = cam_t.overlay_heatmap(raw_img, heatmap_t, alpha=0.5)
        cam_t.remove_hooks()

        # Grad-CAM Scratch
        cam_s = GradCAM(model_scratch, model_scratch.layer4[-1])
        heatmap_s = cam_s.generate_heatmap(tensor, target_class=case["pred_scratch"])
        overlay_s = cam_s.overlay_heatmap(raw_img, heatmap_s, alpha=0.5)
        cam_s.remove_hooks()

        # Consulta ao LMM
        laudo = lmm_service.analyze_radiograph_and_heatmap(
            raw_img,
            overlay_t,
            predicted_class=pred_t_name,
            confidence=conf_t,
        )

        base_filename = f"caso_{idx:02d}_{cat}_{real_name.lower()}"
        img_filename = f"{base_filename}.png"
        laudo_filename = f"{base_filename}_laudo.txt"

        img_save_path = out_path / img_filename
        laudo_save_path = out_path / laudo_filename

        # Salva o laudo
        with open(laudo_save_path, "w", encoding="utf-8") as f:
            f.write(
                f"AUDITORIA CLÍNICA - CASO #{idx:02d} [{cat_descriptions[cat]}]\n"
                f"Arquivo Original: {case['img_path'].name}\n"
                f"Diagnóstico Real: {real_name}\n"
                f"Predição Transfer Learning: {pred_t_name} ({conf_t * 100:.2f}%)\n"
                f"Predição Scratch (Warmup):   {pred_s_name} ({conf_s * 100:.2f}%)\n"
                f"{'=' * 70}\n\n"
                f"{laudo}\n"
            )

        # Renderiza a figura
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

        axes[0].imshow(raw_img.resize((224, 224)), cmap="gray")
        axes[0].set_title(
            f"(A) Radiografia Original\nReal: {real_name} [{cat}]",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].axis("off")

        color_t = "#0d47a1" if case["pred_transfer"] == case["target"] else "#b71c1c"
        axes[1].imshow(overlay_t)
        axes[1].set_title(
            f"(B) Grad-CAM: Transfer Learning\nPred: {pred_t_name} ({conf_t * 100:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color=color_t,
        )
        axes[1].axis("off")

        color_s = "#0d47a1" if case["pred_scratch"] == case["target"] else "#b71c1c"
        axes[2].imshow(overlay_s)
        axes[2].set_title(
            f"(C) Grad-CAM: Scratch (Warmup)\nPred: {pred_s_name} ({conf_s * 100:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color=color_s,
        )
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(img_save_path, bbox_inches="tight")
        plt.close(fig)

        gallery_metadata.append({
            "idx": idx,
            "category": cat,
            "category_desc": cat_descriptions[cat],
            "real_name": real_name,
            "pred_transfer": pred_t_name,
            "conf_transfer": conf_t,
            "pred_scratch": pred_s_name,
            "conf_scratch": conf_s,
            "img_file": img_filename,
            "laudo_file": laudo_filename,
            "source_file": case["img_path"].name,
        })

    # 5. Gera o índice Markdown catalogando toda a galeria
    index_path = out_path / "README.md"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# Galeria de Auditoria Clínica e Explicabilidade Visual (XAI + LMM)\n\n")
        f.write("Esta pasta reúne instâncias radiológicas do conjunto de teste avaliadas com mapas Grad-CAM e laudos estruturados gerados pelo LMM local.\n\n")
        f.write("| Caso | Categoria Clínica | Diagnóstico Real | Pred. Transfer | Pred. Scratch | Painel Visual | Laudo Clínico |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for m in gallery_metadata:
            status_t = "✓" if m["pred_transfer"] == m["real_name"] else "✗"
            f.write(
                f"| #{m['idx']:02d} | **{m['category']}** ({m['category_desc'].split('(')[0].strip()}) | "
                f"`{m['real_name']}` | `{m['pred_transfer']}` ({m['conf_transfer']*100:.1f}%) {status_t} | "
                f"`{m['pred_scratch']}` ({m['conf_scratch']*100:.1f}%) | "
                f"[{m['img_file']}]({m['img_file']}) | [{m['laudo_file']}]({m['laudo_file']}) |\n"
            )
        f.write("\n---\n*Gerado automaticamente pelo serviço `src.plots.gallery_generator`.*\n")

    logger.info(f"Galeria clínica com {len(gallery_metadata)} casos salva em: {out_path}")
    return out_path
