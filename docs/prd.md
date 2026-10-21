# Documento de Requisitos de Produto (PRD) — Avaliação 2 (Visão Computacional)

## 1. Contexto e Requisitos Inegociáveis do Professor

1. **Princípio do Isolamento de Variável (*Ceteris Paribus*):**
   * Manter estritamente fixas as **Técnicas A** e **B**.
   * Variar única e exclusivamente a **Técnica C** (Cenário 1: Proposto) versus **Técnica C Alternativa** (Cenário 2: Contrafactual).
2. **Amostragem Pareada ($N = 30$):**
   * Rodar exatamente 30 pares de experimentos ($60$ treinamentos completos).
   * Conjunto fixo de sementes determinísticas: $\mathcal{S} = \{1001, 1002, \dots, 1030\}$.
   * Em cada semente $s_i$, o sorteio de lotes e o data augmentation estocástico são idênticos para ambos os cenários.
   * O conjunto de teste (582 imagens) permanece estritamente imutável em todas as 60 rodadas.
3. **Inferência Estatística Formal:**
   * Teste $t$ de Student pareado bilateral ($\alpha = 0{,}05$).
   * $H_0: \mu_1 - \mu_2 = 0$ versus $H_1: \mu_1 - \mu_2 \neq 0$.
4. **Entregáveis:**
   * Código modular limpo e reprodutível via CLI (`src/cli.py`).
   * Artigo científico no padrão IEEE de 2 colunas com 4 páginas (`paper/artigo.pdf`).
   * Apresentação estruturada de slides (`slides/apresentacao.md`).
   * Painéis de explicabilidade visual Grad-CAM e laudos estruturados via LMM local (`results/gallery/`).

---

## 2. Escolha Metodológica e Mapeamento das Técnicas

### Caso de Uso Clínico:
* **Dataset:** *Chest X-Ray Dataset* (Kermany et al., 2018).
* **Treinamento:** 4.077 radiografias (1.104 normais e 2.973 pneumonia).
* **Teste Imutável:** 582 radiografias (171 normais e 411 pneumonia).
* **Total:** 4.659 radiografias clínicas reais de tórax pediátrico.

### Mapeamento das Técnicas:

| Componente | Técnica Escolhida | Papel na Metodologia |
| :--- | :--- | :--- |
| **Técnica A (Fixa / Pré-processamento)** | **Data Augmentation Estocástico** | Redimensionamento para $224 \times 224$ px, espelhamento horizontal estocástico ($p=0.5$), rotação aleatória ($\pm 10^\circ$), jitter fotométrico e normalização ImageNet. Semente $s_i$ governa a geração estocástica. |
| **Técnica B (Fixa / Explicabilidade & Auditoria)** | **ResNet-50 + Grad-CAM + LMM Local (Gemma 4)** | Arquitetura convolucional profunda com conexões residuais, extrator de mapas de ativação Grad-CAM na camada `layer4`, e auditoria diagnóstica clínica por LMM via interface OpenAI-compatible (`localhost:8080`). |
| **Técnica C (Cenário 1 - Proposto)** | **Transfer Learning (Pesos ImageNet-1k)** | ResNet-50 inicializada com pesos convolucionais consolidados da ImageNet, otimizada com Adam ($\eta = 10^{-4}$, decaimento $L_2 = 10^{-4}$). |
| **Técnica C Alternativa (Cenário 2 - Contrafactual)** | **Treinamento do Zero (Scratch + Warmup/Cosine)** | ResNet-50 inicializada pelo método He Normal (Kaiming), submetida a 1 época de *Linear Warmup* seguida por decaimento suave por cosseno (*Cosine Annealing*) com taxa de aprendizado variando de $3 \times 10^{-4}$ a $10^{-6}$. |

---

## 3. Protocolo Experimental e Resultados Empíricos Reais

### 1. Teste t de Student Pareado ($N=30$, gl=29):
* **F1-Macro:** Diferença média $\bar{d} = +0{,}0997$, $s_d = 0{,}0542$, $t = 10{,}07$, $p = 5{,}63 \times 10^{-11}$, $\text{IC}_{95\%} = [0{,}0794, 0{,}1199]$.
* **Acurácia:** Diferença média $\bar{d} = +0{,}0918$, $s_d = 0{,}0569$, $t = 8{,}83$, $p = 1{,}03 \times 10^{-09}$, $\text{IC}_{95\%} = [0{,}0705, 0{,}1130]$.
* **Decisão Estatística:** Rejeição inequívoca de $H_0$ em favor de $H_1$.

### 2. Decomposição Amostral e Matriz de Confusão (Semente 1001, $N_{teste}=582$):
* **Cenário 1 (Transfer Learning):**
  * Acertos: **537 de 582 imagens ($92{,}27\%$)** | Erros: **45 de 582 imagens ($7{,}73\%$)**
  * Sensibilidade (Pneumonia): **$99{,}27\%$** ($408$ de $411$ detectadas, apenas 3 falsos negativos).
  * Especificidade (Sadio): **$75{,}44\%$** ($129$ de $171$ confirmadas sadias).
* **Cenário 2 (Scratch + Warmup):**
  * Acertos: **500 de 582 imagens ($85{,}91\%$)** | Erros: **82 de 582 imagens ($14{,}09\%$)**
  * Sensibilidade (Pneumonia): **$98{,}29\%$** ($404$ de $411$ detectadas, 7 falsos negativos).
  * Especificidade (Sadio): **$56{,}14\%$** ($96$ de $171$ confirmadas sadias, 75 falsos positivos).
