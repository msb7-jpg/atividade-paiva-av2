# Plano de Implementação: Transição para Treinamento e Avaliação Empírica Real (Beans Dataset + ViT)

Este documento formaliza o plano de transição da codebase para execução **100% empírica e real**, utilizando o **Beans Dataset** e o modelo **Vision Transformer (ViT-B/32 - CLIP)**, eliminando quaisquer dados sintéticos ou aproximados, preservando o rigor científico (*Ceteris Paribus*, $N=30$, pareamento de sementes).

---

## 1. Contexto e Diagnóstico

- O pipeline em [`src/services/experiment_runner.py`](../src/services/experiment_runner.py) atualmente simulava as métricas através de distribuições normais (`rng.normal`).
- O objetivo desta transição é fazer com que o script oficial (`./scripts/run_simulation.sh` e `src/cli.py`) execute **por padrão sempre os modelos reais e os dados reais**, sem necessidade de flags adicionais como `--real`.
- A metodologia e as especificações formais definidas em [`docs/especificacao-metodologica.md`](especificacao-metodologica.md) e [`docs/prd.md`](prd.md) serão estritamente respeitadas:
  - **Dataset:** *Beans Dataset* (1.034 imagens de treino, 133 validação, 128 teste; 3 classes).
  - **Técnica A (Fixa):** Extração de representações vetoriais de 512 dimensões via Vision Transformer (`ViT-B/32`).
  - **Técnica B (Fixa):** Cabeça densa com Dropout ($p=0.20$) e regularização Ridge ($L_2 = 10^{-4}$).
  - **Técnica C (Cenário 1):** Transfer Learning com pesos pré-treinados multimodais do CLIP (Linear Probe).
  - **Técnica C Alternativa (Cenário 2):** Treinamento a partir do zero com pesos aleatórios Gaussianos (Scratch).
  - **Amostragem Pareada:** 30 sementes fixas $\mathcal{S} = \{1001, 1002, \dots, 1030\}$.

---

## 2. Estratégia de Engenharia e Eficiência

Para garantir que o experimento rode de forma agnóstica em qualquer máquina (com GPU CUDA disponível ou apenas CPU, sem assumir hardware local específico):
1. **Carregamento Automático do Dataset:** Download e preparação automática do *Beans Dataset* via biblioteca `datasets` do Hugging Face.
2. **Pré-extração Eficiente de Embeddings:**
   - Passagem única do conjunto de treino e teste pelo backbone do ViT para extrair os embeddings densos de 512 dimensões.
   - Cenário 1: Embeddings gerados com pesos pré-treinados do CLIP (`openai/clip-vit-base-patch32`).
   - Cenário 2: Embeddings gerados com backbone inicializado aleatoriamente (`CLIPVisionModelWithProjection` a partir de `CLIPVisionConfig`).
3. **Treinamento Supervisionado Pareado ($N=30$):**
   - Execução direta dos 30 pares de treinamento da cabeça linear (Linear Probe) em PyTorch com Adam, Dropout e $L_2$.
   - Avaliação direta no conjunto de teste independente (128 amostras reais).
   - Cálculo das métricas empíricas exatas ($F_1$-Macro e Acurácia) com `scikit-learn` a cada iteração.
   - Execução ultrarrápida e portável em qualquer ambiente.

---

## 3. Mapeamento das Mudanças na Codebase

### A. Configuração do Projeto e Dependências
- **[`pyproject.toml`](../pyproject.toml):**
  - Adicionar dependência `"datasets>=2.18.0"`.
  - Configurar `[tool.hatch.build.targets.wheel] packages = ["src"]` para suporte pleno ao `uv pip install -e .`.

### B. Repositório de Dados
- **[`src/repositories/dataset_repository.py`](../src/repositories/dataset_repository.py):**
  - Download e pré-processamento padronizado das imagens ($224 \times 224$, normalização padrão CLIP).
  - Retorno dos tensores de treino e teste com seus respectivos rótulos.

### C. Extração de Características
- **[`src/services/feature_extractor.py`](../src/services/feature_extractor.py):**
  - Carregamento do extrator pré-treinado (Cenário 1) e com pesos aleatórios (Cenário 2).
  - Extração dos vetores latentes com dispositivo automático (`cuda` se disponível, senão `cpu`).

### D. Treinador Experimental e CLI
- **[`src/services/experiment_runner.py`](../src/services/experiment_runner.py):**
  - Substituição da geração sintética pelo loop de treino e inferência real em PyTorch.
  - Salvamento dos resultados empíricos em `results/simulacoes_pareadas.json`.
- **[`src/cli.py`](../src/cli.py):**
  - Execução padrão direta e transparente dos dados e modelos reais.

### E. Artigo Científico e Visualizações
- Reexecução do `./scripts/run_simulation.sh`.
- Recompilação do artigo `./scripts/build_paper.sh` com os resultados estatísticos empíricos obtidos.
- Atualização contínua do [`PROGRESS.md`](../PROGRESS.md).
