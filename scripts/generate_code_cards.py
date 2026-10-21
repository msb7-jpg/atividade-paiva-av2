"""Gerador de Cards Visuais de Código Pedagógico para Apresentação em Slides.

Renderiza trechos Python essenciais com Highlight.js e Chromium Headless (Retina 2x),
salvando imagens PNG em results/code_cards/.
"""

import html
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image

ASSETS_DIR = Path("results/code_cards/assets")
OUTPUT_DIR = Path("results/code_cards")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Download / Cache de assets do Highlight.js
HLJS_JS_PATH = ASSETS_DIR / "highlight.min.js"
HLJS_CSS_PATH = ASSETS_DIR / "atom-one-dark.min.css"

if not HLJS_JS_PATH.exists():
    urllib.request.urlretrieve(
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js",
        HLJS_JS_PATH,
    )

if not HLJS_CSS_PATH.exists():
    urllib.request.urlretrieve(
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css",
        HLJS_CSS_PATH,
    )

HLJS_JS = HLJS_JS_PATH.read_text(encoding="utf-8")
HLJS_CSS = HLJS_CSS_PATH.read_text(encoding="utf-8")

# 2. Definição dos 7 trechos de código pedagógicos
CARDS = [
    {
        "id": "card_01_data_augmentation",
        "title": "Técnica A: Data Augmentation Estocástico com Restrição Anatômica",
        "file": "src/repositories/dataset_repository.py",
        "speaker": "Miguel Batista",
        "takeaway": "A rotação é estritamente limitada a ±10° e o espelhamento é apenas horizontal: inversões verticais criariam radiografias anatomicamente impossíveis.",
        "code": """import torchvision.transforms as T

# -------------------------------------------------------------------------
# Técnica A (Fixa): Data Augmentation com Preservação Anatômica Torácica
# Por que: Aumenta a diversidade de treino (4.077 imagens) sem corromper
# a orientação anatômica necessária para diagnosticar infiltrados pulmonares.
# -------------------------------------------------------------------------
def get_clinical_train_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        # Espelhamento horizontal (50% de chance) preserva simetria torácica bilateral
        T.RandomHorizontalFlip(p=0.5),
        # Leve rotação (±10°) simula pequenas variações de posicionamento do paciente
        T.RandomRotation(degrees=10),
        # Perturbação fotométrica (±10%) simula oscilações de penetração de raios-X
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        # Normalização com parâmetros do ImageNet para manter gradientes estáveis
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])""",
    },
    {
        "id": "card_02_resnet50_ceteris_paribus",
        "title": "Técnica B, C vs C_alt: Isolamento Ceteris Paribus na ResNet-50",
        "file": "src/services/resnet_builder.py",
        "speaker": "Beatriz Pereira",
        "takeaway": "A arquitetura residual e o otimizador são fixados; isolamos estritamente a inicialização pré-treinada (ImageNet) vs estocástica (He Normal).",
        "code": """import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

# -------------------------------------------------------------------------
# Princípio Ceteris Paribus: ResNet-50 com Transfer Learning vs Scratch
# Por que: Conexões de salto F(x)+x evitam dissipação de gradiente. A rede é
# mantida idêntica, isolando a inicialização como variável independente pura.
# -------------------------------------------------------------------------
def build_resnet50_scenario(pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if not pretrained:
        # Cenário 2 (Scratch): Inicialização He Normal (Kaiming) para ReLU
        # Garante variância constante das ativações ao longo das 50 camadas
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # Cabeça linear binária compartilhada: 0 = NORMAL, 1 = PNEUMONIA
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model""",
    },
    {
        "id": "card_03_paired_dataloader",
        "title": "Protocolo Pareado: Isolamento de Geradores no DataLoader",
        "file": "src/services/experiment_runner.py",
        "speaker": "Manuela Cavalcanti",
        "takeaway": "Geradores independentes com a semente s_i garantem que Transfer e Scratch enfrentem rigorosamente a mesma ordem de lotes e transformações.",
        "code": """import torch
from torch.utils.data import DataLoader, Dataset

# -------------------------------------------------------------------------
# Protocolo Pareado Estrito: Isolamento de Geradores Aleatórios por Semente
# Por que: Sem gerador dedicado, o Scratch herda o gerador já avançado do
# Transfer, quebrando a premissa de condições estocásticas rigorosamente iguais.
# -------------------------------------------------------------------------
def create_paired_dataloaders(dataset: Dataset, seed: int, batch_size: int = 64):
    def make_loader() -> DataLoader:
        # Gerador isolado semeado: garante paridade exata de permutações
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=4,
            pin_memory=True,
        )

    # Ambos os cenários recebem exatamente os mesmos batches e augmentations
    loader_transfer = make_loader()
    loader_scratch = make_loader()
    return loader_transfer, loader_scratch""",
    },
    {
        "id": "card_04_paired_t_test",
        "title": "Inferência Estatística Formal: Teste t de Student Pareado",
        "file": "src/services/statistical_service.py",
        "speaker": "Rafael Barros",
        "takeaway": "A diferença d_i = F1_trans - F1_scratch cancela o ruído comum da semente; N=30 viabiliza o Teorema Central do Limite e rejeita H0 (p=5.63e-11).",
        "code": """import numpy as np
from scipy import stats

# -------------------------------------------------------------------------
# Inferência Estatística Formal: Teste t Pareado Bilateral (N=30, gl=29)
# Por que: Avalia se a superioridade de Transfer Learning é estatisticamente
# real ou fruto de mero ruído amostral, calculando o tamanho de efeito.
# -------------------------------------------------------------------------
def compute_paired_statistics(f1_transfer: list[float], f1_scratch: list[float]):
    d = np.array(f1_transfer) - np.array(f1_scratch)  # Diferenças pareadas
    n = len(d)                                       # n = 30 sementes
    mean_diff = float(np.mean(d))                    # d_barra = +0.1015 (+10.15%)
    std_diff = float(np.std(d, ddof=1))              # Desvio padrão das diferenças
    se_diff = std_diff / np.sqrt(n)                  # Erro padrão da média

    t_stat = mean_diff / se_diff                     # t = 11.23
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))  # p = 5.63e-11
    cohens_d = mean_diff / std_diff                  # d de Cohen = 2.05 (Efeito Enorme)

    # Rejeição categórica da Hipótese Nula (p << 0.05) em prol de H1
    return {"mean_diff": mean_diff, "t": t_stat, "p_value": p_value, "cohens_d": cohens_d}""",
    },
    {
        "id": "card_05_confusion_matrix_audit",
        "title": "Auditoria Diagnóstica Amostral: Matriz de Confusão & Trade-offs",
        "file": "src/cli.py & src/domain/models.py",
        "speaker": "Ygor Rosa",
        "takeaway": "Na clínica pediátrica, Falsos Negativos (FN) são inaceitáveis. O Transfer alcança 99.27% de sensibilidade com apenas 3 FNs, contra 7 do Scratch.",
        "code": """# -------------------------------------------------------------------------
# Auditoria Diagnóstica Clínica: Avaliação nas 582 Radiografias de Teste
# Por que: A acurácia isolada mascara desbalanceamentos; sensibilidade mede
# a detecção da pneumonia e especificidade protege contra alarmes falsos.
# -------------------------------------------------------------------------
def audit_clinical_performance(tp: int, tn: int, fp: int, fn: int):
    total = tp + tn + fp + fn                        # 582 imagens de teste
    accuracy = (tp + tn) / total                     # Acurácia global
    sensitivity = tp / (tp + fn)                     # Sensibilidade (Pneumonia: TP / 411)
    specificity = tn / (tn + fp)                     # Especificidade (Sadio: TN / 171)

    # Transfer Learning (Semente 1001): 537/582 acertos (92.27%), apenas 3 FNs!
    # Scratch (Semente 1001):           500/582 acertos (85.91%), colapso com 75 FPs!
    return {
        "acertos_totais": f"{tp + tn} / {total} ({accuracy:.2%})",
        "sensibilidade_pneumonia": f"{sensitivity:.2%} ({fn} Falsos Negativos)",
        "especificidade_sadio": f"{specificity:.2%} ({fp} Falsos Positivos)",
    }""",
    },
    {
        "id": "card_06_gradcam_pulmonary",
        "title": "Interpretabilidade Visual: Grad-CAM na layer4 da ResNet-50",
        "file": "src/services/gradcam_service.py",
        "speaker": "Hugo Alcântara",
        "takeaway": "O filtro ReLU isola estritamente as feições com gradiente positivo; pesos ponderados evidenciam onde a rede enxergou consolidações pulmonares.",
        "code": """import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Explicabilidade Visual Grad-CAM na última camada convolucional (layer4)
# Por que: Verifica se a decisão da ResNet-50 está ancorada na anatomia
# pulmonar ou se o modelo está sendo induzido por artefatos espúrios de borda.
# -------------------------------------------------------------------------
def compute_gradcam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    # 1. Global Average Pooling sobre os gradientes espaciais -> pesos alfa_k
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, 2048, 1, 1]

    # 2. Combinação linear dos canais ponderados pela importância na predição
    cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

    # 3. Retificação ReLU: mantém apenas ativações que aumentam a nota da classe
    cam = F.relu(cam)

    # 4. Normalização min-max para sobreposição como mapa de calor térmico
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.squeeze()""",
    },
    {
        "id": "card_07_lmm_insights",
        "title": "Auditoria Diagnóstica Multimodal: LMM Local com Fallback Estrito",
        "file": "src/services/lmm_insights_service.py",
        "speaker": "Rafaela Vidal",
        "takeaway": "Validação automática por LMM (Gemma 4 local em localhost:8080) atestando coerência anatômica, falhando explicitamente em caso de indisponibilidade.",
        "code": '''import httpx

# -------------------------------------------------------------------------
# Auditoria Multimodal via LMM Local (Gemma 4 no llama-server em :8080)
# Por que: Automatiza a verificação clínica do mapa de calor Grad-CAM,
# garantindo que laudos nunca sejam simulados se o serviço estiver offline.
# -------------------------------------------------------------------------
def audit_heatmap_with_lmm(original_b64: str, heatmap_b64: str, pred_class: str) -> str:
    prompt = (
        f"Avalie a coerência anatômica do Grad-CAM para a predição '{pred_class}'. "
        "O foco está no parênquima pulmonar ou em artefatos ósseos/bordas?"
    )
    try:
        response = httpx.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            json={
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{heatmap_b64}"}},
                    ],
                }],
                "max_tokens": 300,
            },
            timeout=30.0,
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as err:
        # Falha explícita com aviso transparente (eliminação de mocks enganosos)
        return f"[AVISO: LAUDO NÃO GERADO] Servidor LMM indisponível: {err}"''',
    },
]

# 3. Template HTML com estilo Ray.so / Carbon
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  {hljs_css}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 24px;
    width: fit-content;
    height: fit-content;
  }}

  .card-container {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
    width: 980px;
    overflow: hidden;
  }}

  .card-header {{
    background: rgba(15, 23, 42, 0.85);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}

  .window-dots {{
    display: flex;
    gap: 8px;
  }}

  .dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }}
  .dot-red {{ background: #ef4444; }}
  .dot-yellow {{ background: #f59e0b; }}
  .dot-green {{ background: #10b981; }}

  .card-meta {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}

  .file-badge {{
    background: rgba(255, 255, 255, 0.08);
    color: #94a3b8;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.06);
  }}

  .speaker-badge {{
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid rgba(56, 189, 248, 0.25);
  }}

  .card-title-bar {{
    padding: 12px 24px 6px 24px;
    background: #282c34;
  }}

  .card-title {{
    color: #f1f5f9;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.2px;
  }}

  .code-area {{
    background: #282c34;
    padding: 10px 24px 20px 24px;
  }}

  pre {{
    margin: 0;
  }}

  code.hljs {{
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace !important;
    font-size: 13.5px !important;
    line-height: 1.55 !important;
    background: transparent !important;
    padding: 0 !important;
  }}

  .card-footer {{
    background: rgba(15, 23, 42, 0.95);
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding: 12px 20px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }}

  .footer-icon {{
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    font-weight: bold;
    font-size: 11px;
    padding: 2px 7px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 1px;
    flex-shrink: 0;
  }}

  .footer-text {{
    color: #cbd5e1;
    font-size: 12.5px;
    line-height: 1.45;
  }}
</style>
</head>
<body>

<div class="card-container" id="capture-target">
  <div class="card-header">
    <div class="window-dots">
      <div class="dot dot-red"></div>
      <div class="dot dot-yellow"></div>
      <div class="dot dot-green"></div>
    </div>
    <div class="card-meta">
      <span class="speaker-badge">{speaker}</span>
      <span class="file-badge">{file}</span>
    </div>
  </div>

  <div class="card-title-bar">
    <div class="card-title">{title}</div>
  </div>

  <div class="code-area">
    <pre><code class="language-python">{code}</code></pre>
  </div>

  <div class="card-footer">
    <span class="footer-icon">Por que</span>
    <span class="footer-text">{takeaway}</span>
  </div>
</div>

<script>{hljs_js}</script>
<script>hljs.highlightAll();</script>
</body>
</html>
"""


def render_card(card: dict) -> Path:
    html_content = HTML_TEMPLATE.format(
        hljs_css=HLJS_CSS,
        hljs_js=HLJS_JS,
        title=html.escape(card["title"]),
        file=html.escape(card["file"]),
        speaker=html.escape(card["speaker"]),
        takeaway=html.escape(card["takeaway"]),
        code=html.escape(card["code"]),
    )

    temp_html = ASSETS_DIR / f"{card['id']}.html"
    temp_html.write_text(html_content, encoding="utf-8")

    out_png = OUTPUT_DIR / f"{card['id']}.png"

    # Captura com Chromium headless com 2x scale para qualidade Retina e fundo transparente
    cmd = [
        "/usr/bin/chromium",
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        "--default-background-color=00000000",
        "--force-device-scale-factor=2",
        "--window-size=1080,1400",
        f"--screenshot={out_png}",
        f"file://{temp_html.resolve()}",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Corta o fundo transparente ao redor do container preservando as sombras suaves
    with Image.open(out_png) as img:
        bbox = img.getbbox()
        if bbox:
            pad = 16
            crop_box = (
                max(0, bbox[0] - pad),
                max(0, bbox[1] - pad),
                min(img.width, bbox[2] + pad),
                min(img.height, bbox[3] + pad),
            )
            cropped = img.crop(crop_box)
            cropped.save(out_png, "PNG")

    temp_html.unlink(missing_ok=True)
    return out_png


def main():
    print(f"==> Gerando {len(CARDS)} cards de código pedagógico...")
    for idx, card in enumerate(CARDS, start=1):
        png_path = render_card(card)
        print(f"[{idx}/{len(CARDS)}] ✓ Salvo: {png_path} ({card['speaker']}: {card['title']})")
    print(f"\n==> Concluído com sucesso! Imagens prontas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
