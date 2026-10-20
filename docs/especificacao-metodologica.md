# Especificação Metodológica do Estudo Comparativo

Este documento estabelece o protocolo científico formal para a **2ª Avaliação da disciplina de Inteligência Artificial** (Prof. Rodrigo Paiva - CESAR School).

---

## 1. Formulação do Problema e Hipóteses

O experimento é conduzido sob a metodologia *ceteris paribus*, isolando uma única variável independente para mensuração de causalidade em aprendizado profundo.

- **Variável Dependente (Métricas):** $F_1\text{-Macro}$ e Acurácia no conjunto de teste independente.
- **Variável Independente:** Origem dos pesos do extrator de características visual (pesos pré-treinados em larga escala multimodal vs. inicialização aleatória).
- **Variáveis de Controle:** Arquitetura do extrator, arquitetura da cabeça classificadora, taxa de aprendizado, algoritmo de otimização, hiperparâmetros de regularização e sementes aleatórias pareadas.

### Formulação Matemática das Hipóteses:

Sejam $\mu_1$ e $\mu_2$ os desempenhos médios populacionais no conjunto de teste do **Cenário 1** (Transfer Learning) e **Cenário 2** (Pesos Aleatórios), respectivamente:

- **Hipótese Nula ($H_0$):** 
  $$\mu_1 - \mu_2 = 0$$
  *Afirma que não existe diferença estatisticamente significativa na eficácia entre inicializar com pesos pré-treinados ou pesos aleatórios; quaisquer divergências observadas nas amostras decorrem puramente de variabilidade estocástica.*

- **Hipótese Alternativa ($H_1$):** 
  $$\mu_1 - \mu_2 \neq 0$$
  *Afirma que existe diferença estatisticamente significativa na eficácia média entre as duas abordagens ao nível de significância de $\alpha = 0{,}05$.*

---

## 2. Seleção e Mapeamento Obrigatório das Técnicas da Aula

| Componente | Técnica Escolhida | Papel no Pipeline Científico |
| :--- | :--- | :--- |
| **Técnica A (Fixa)** | **Extração de Representações Visuais via Vision Transformer / CLIP** (`ViT-B/32`) | Converte as imagens redimensionadas ($224 \times 224$ pixels) em representações vetoriais latentes ($512$ dimensões), preservando relações espaciais e semânticas globais através de auto-atenção. |
| **Técnica B (Fixa)** | **Prevenção de Overfitting via Dropout ($p=0.2$) + Regularização Ridge ($L_2$, $\lambda = 10^{-4}$)** | Inserida na cabeça classificadora supervisionada para inibir co-adaptações de neurônios e penalizar a magnitude de pesos em amostras pequenas. |
| **Técnica C (Cenário 1)** | **Transfer Learning Supervisionado (Linear Probe)** | Utiliza o extrator de características com pesos pré-treinados pelo CLIP em 400 milhões de pares (imagem-texto), ajustando a camada linear final. |
| **Técnica C Alternativa (Cenário 2)** | **Treinamento a Partir do Zero (Pesos Aleatórios / Scratch)** | Utiliza o mesmo extrator com inicialização aleatória Gaussiana truncated/Xavier, forçando a rede a aprender representações apenas com os dados locais do dataset. |

---

## 3. Protocolo de Controle Experimental Pareado ($N=30$)

1. **Amostragem Pareada:**
   - São executadas 30 simulações pareadas (total de 60 treinamentos completos).
   - O conjunto de sementes estocásticas compreende o intervalo inteiro:
     $$\mathcal{S} = \{1001, 1002, 1003, \dots, 1030\}$$
   - Em cada iteração $i \in \{1, \dots, 30\}$ com semente $s_i \in \mathcal{S}$:
     - A semente do PyTorch, NumPy e Python RNG é fixada estritamente em $s_i$.
     - O particionamento de batches, embaralhamento e augmentations são idênticos para ambos os cenários na rodada $i$.
     - O conjunto de teste (128 imagens no Beans Dataset) permanece imutável e idêntico em todas as 60 execuções.

2. **Inferência Estatística (Teste $t$ Pareado Bilateral):**
   - Para cada par $i$, calcula-se a diferença pareada:
     $$d_i = X_{1, i} - X_{2, i}$$
   - Estatística do teste:
     $$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$
     onde $\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i$, $s_d$ é o desvio padrão amostral das diferenças e $n = 30$ ($29$ graus de liberdade).
   - Intervalo de Confiança ($95\%$):
     $$\text{IC}_{95\%} = \left[ \bar{d} - t_{0.025, 29} \frac{s_d}{\sqrt{n}}, \quad \bar{d} + t_{0.025, 29} \frac{s_d}{\sqrt{n}} \right]$$
   - **Critério de Decisão:** Rejeita-se $H_0$ se $p\text{-valor} < 0{,}05$.

3. **Explicabilidade Visual (XAI):**
   - Extração de mapas de relevância e atenção em 3 instâncias de teste fixas (uma de cada classe patológica: *Angular Leaf Spot*, *Bean Rust*, *Healthy*).
   - Comparação da focalização espacial entre a rede com pesos pré-treinados e a rede com pesos aleatórios.
