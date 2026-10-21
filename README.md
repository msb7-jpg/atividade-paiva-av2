# Estudo Comparativo de Otimização em CNNs para Diagnóstico de Pneumonia: Transferência de Aprendizado vs. Treinamento do Zero

Repositório oficial do trabalho da **2ª Avaliação da disciplina de Inteligência Artificial (EL1)** — **CESAR School** (Recife, PE, Brasil).  
**Docente:** Prof. Rodrigo Paiva (`rfapo@cesar.school`)  

### Autores
- **Miguel Batista** (`msb2@cesar.school`)
- **Rafaela Vidal** (`rbv2@cesar.school`)
- **Beatriz Pereira** (`bcbsp@cesar.school`)
- **Manuela Cavalcanti** (`mca2@cesar.school`)
- **Rafael Barros** (`ralb@cesar.school`)

---

## 1. Visão Geral da Pesquisa

Este estudo conduz uma investigação científica controlada e reprodutível sobre a eficácia da **Transferência de Aprendizado (*Transfer Learning*)** frente ao **Treinamento a partir do Zero (*Scratch*)** com *Linear Warmup* e decaimento por cosseno (*Cosine Annealing*), isolando estritamente a variável independente sob o princípio *ceteris paribus*.

```
=============================================================================
                      METODOLOGIA CETERIS PARIBUS
=============================================================================
TÉCNICA A (Fixa / Pré-processamento):
  • Data Augmentation estocástico (Resize 224x224, Flip Horizontal, Rotação, Jitter)
  • Semente s_i fixa governando as transformações

TÉCNICA B (Fixa / Explicabilidade & Auditoria):
  • Arquitetura profunda ResNet-50 com Grad-CAM (layer4)
  • Parecer clínico automatizado via LMM multimodal local (Gemma 4 em localhost:8080)

TÉCNICA C (Variável Independente):
  • Cenário 1 (Proposto): ResNet-50 com pesos pré-treinados ImageNet-1k, Adam (lr=1e-4)
  • Cenário 2 (Contrafactual): ResNet-50 He Normal do zero + Warmup/Cosine Annealing
=============================================================================
```

---

## 2. Resultados Empíricos Principais

### A. Inferência Estatística Formal (Teste t Pareado Bilateral, $N=30$, gl=29)

| Métrica | Diferença Média ($\bar{d}$) | Desvio ($s_d$) | Estatística $t$ | Valor-$p$ | IC 95% | Decisão ($\alpha=0{,}05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1-Macro** | **$+0{,}0997$** | $0{,}0542$ | **$10{,}07$** | **$5{,}63 \times 10^{-11}$** | $[+0{,}0794, +0{,}1199]$ | **Rejeita $H_0$** |
| **Acurácia** | **$+0{,}0918$** | $0{,}0569$ | **$8{,}83$** | **$1{,}03 \times 10^{-09}$** | $[+0{,}0705, +0{,}1130]$ | **Rejeita $H_0$** |

### B. Decomposição Amostral e Matriz de Confusão no Teste ($N=582$)

| Métrica Diagnóstica | Transfer Learning (ImageNet) | Scratch (Warmup/Cosine) | Impacto / Diferença |
| :--- | :---: | :---: | :---: |
| **Classificações Corretas** | **537 / 582 ($92{,}27\%$)** | 500 / 582 ($85{,}91\%$) | **$+37$ acertos ($+6{,}36\%$)** |
| **Falhas / Erros Globais** | **45 / 582 ($7{,}73\%$)** | 82 / 582 ($14{,}09\%$) | **$-37$ falhas ($-45{,}12\%$)** |
| **Verdadeiros Positivos (TP)** | **408** ($99{,}27\%$) | 404 ($98{,}29\%$) | $+4$ detecções |
| **Verdadeiros Negativos (TN)** | **129** ($75{,}44\%$) | 96 ($56{,}14\%$) | $+33$ confirmações sadias |
| **Falsos Positivos (FP)** | **42** ($24{,}56\%$) | 75 ($43{,}86\%$) | $-33$ falsos alarmes |
| **Falsos Negativos (FN)** | **3** ($0{,}73\%$) | 7 ($1{,}71\%$) | **Apenas 3 casos críticos perdidos** |
| **Sensibilidade (Recall)** | **$99{,}27\%$** | $98{,}29\%$ | $+0{,}98\%$ |
| **Especificidade** | **$75{,}44\%$** | $56{,}14\%$ | **$+19{,}30\%$** |

---

## 3. Estrutura do Repositório

```
paiva-av2/
├── docs/                # Especificações metodológicas formais e PRD
├── latex/               # Template oficial IEEEtran, referências BibTeX e CSL
├── paper/               # Artigo científico IEEE (artigo.md e artigo.pdf de 4 páginas)
│   └── pages/           # Imagens PNG de cada página do artigo compilado
├── results/             # Gráficos, relatórios e saídas persistidas
│   ├── runs/            # Métricas individuais salvas por semente (seed_1001.json, etc.)
│   ├── checkpoints/     # Checkpoints PyTorch dos modelos treinados (.pt)
│   ├── gallery/         # Galeria clínica multi-casos (TP, TN, FP, FN) com laudos LMM
│   ├── boxplot_comparativo.png
│   ├── explicabilidade_xai.png
│   └── fluxograma_metodologia.png
├── scripts/             # Scripts utilitários de execução e compilação
│   ├── build_paper.sh   # Compilação do artigo Pandoc/LaTeX em PDF
│   └── run_simulation.sh# Execução das 30 simulações pareadas
└── src/                 # Codebase modular em Clean Architecture
    ├── domain/          # Modelos Pydantic imutáveis
    ├── repositories/    # Dataset loaders e transformações
    ├── services/        # Orquestrador de treino, teste t e lifecycle LMM
    ├── plots/           # Geradores de gráficos, XAI e galeria
    └── cli.py           # Interface de Linha de Comando rica (Typer & Rich)
```

---

## 4. Guia de Uso da Linha de Comando (CLI)

O projeto é gerenciado via `uv` (Python 3.12+):

### 1. Inspecionar o Status das Simulações e Checkpoints
```bash
uv run python -m src.cli status
```

### 2. Executar as 30 Simulações Pareadas (com Retomada Inteligente)
```bash
uv run python -m src.cli run --resume
```

### 3. Executar Auditoria XAI e Laudo do LMM Isoladamente
```bash
uv run python -m src.cli xai --seed 1001
```

### 4. Gerar Galeria Clínica Multi-Casos com Laudos
```bash
uv run python -m src.cli gallery --samples 2 --seed 1001
```
Gera painéis e laudos em `results/gallery/` para Verdadeiros Positivos, Sadios e casos de erro.

### 5. Regenerar Diagrama Metodológico Oficial
```bash
uv run python -m src.cli diagram
```

### 6. Compilar o Artigo Científico IEEE em PDF
```bash
./scripts/build_paper.sh
```
O artigo é compilado em [`paper/artigo.pdf`](paper/artigo.pdf) e renderizado para inspeção visual em [`paper/pages/`](paper/pages/).
