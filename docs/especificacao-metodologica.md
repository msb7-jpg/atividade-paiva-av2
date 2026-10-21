# Especificação Metodológica do Estudo Comparativo

Este documento estabelece o protocolo científico formal para a **2ª Avaliação da disciplina de Inteligência Artificial** (Prof. Rodrigo Paiva - CESAR School).

---

## 1. Formulação do Problema e Hipóteses

O experimento é conduzido sob a metodologia *ceteris paribus*, isolando uma única variável independente para mensuração de causalidade em aprendizado profundo aplicado a diagnóstico por imagem médica.

- **Variável Dependente (Métricas):** $F_1\text{-Macro}$, Acurácia, Sensibilidade e Especificidade no conjunto de teste independente ($N_{teste} = 582$).
- **Variável Independente:** Estratégia de inicialização e otimização dos pesos convolucionais da ResNet-50 (Transferência de Aprendizado com pesos pré-treinados ImageNet-1k vs. Treinamento do zero com He Normal e agendamento Linear Warmup + Cosine Annealing).
- **Variáveis de Controle:** Arquitetura ResNet-50, hiperparâmetros de Data Augmentation estocástico, lote ($batch = 64$), regularização $L_2$ ($10^{-4}$), dispositivo GPU acelerado e sementes aleatórias pareadas.

### Formulação Matemática das Hipóteses:

Sejam $\mu_1$ e $\mu_2$ os desempenhos médios populacionais no conjunto de teste do **Cenário 1** (Transfer Learning) e **Cenário 2** (Scratch), respectivamente:

- **Hipótese Nula ($H_0$):** 
  $$\mu_1 - \mu_2 = 0$$
  *Afirma que não existe diferença estatisticamente significativa na eficácia diagnóstica entre inicializar com pesos pré-treinados ou treinar a partir do zero; quaisquer divergências observadas nas amostras decorrem puramente de variabilidade estocástica.*

- **Hipótese Alternativa ($H_1$):** 
  $$\mu_1 - \mu_2 \neq 0$$
  *Afirma que existe diferença estatisticamente significativa na eficácia média entre as duas abordagens ao nível de significância de $\alpha = 0{,}05$.*

---

## 2. Seleção e Mapeamento das Técnicas

| Componente | Técnica Escolhida | Papel no Pipeline Científico |
| :--- | :--- | :--- |
| **Técnica A (Fixa / Pré-processamento)** | **Data Augmentation Estocástico** | Redimensionamento para $224 \times 224$ px, espelhamento horizontal ($p=0.5$), rotação aleatória ($\pm 10^\circ$), Color Jitter fotométrico e normalização ImageNet. Governação determinística pela semente $s_i$. |
| **Técnica B (Fixa / Arquitetura e Explicabilidade)** | **ResNet-50 + Grad-CAM (`layer4`) + LMM Local (Gemma 4)** | Modelo profundo com conexões residuais de salto. Extração de mapas de calor Grad-CAM na última camada convolucional e geração automatizada de parecer clínico via LMM em `localhost:8080`. |
| **Técnica C (Cenário 1 - Proposto)** | **Transfer Learning (ImageNet-1k)** | Pesos convolucionais pré-treinados no ImageNet-1k com otimizador Adam ($\eta = 10^{-4}$, weight decay $L_2 = 10^{-4}$). |
| **Técnica C Alternativa (Cenário 2 - Contrafactual)** | **Treinamento do Zero (He Normal + Cosine)** | Pesos inicializados do zero por He Normal (Kaiming), submetidos a 1 época de *Linear Warmup* e decaimento por cosseno ($\eta \in [10^{-6}, 3 \times 10^{-4}]$). |

---

## 3. Protocolo de Controle Experimental Pareado ($N=30$)

1. **Amostragem Pareada:**
   - 30 simulações pareadas (total de 60 treinamentos completos).
   - Conjunto fixo de sementes: $\mathcal{S} = \{1001, 1002, 1003, \dots, 1030\}$.
   - Em cada iteração $i \in \{1, \dots, 30\}$ com semente $s_i \in \mathcal{S}$:
     - A semente do PyTorch, NumPy e Python RNG é fixada estritamente em $s_i$.
     - O particionamento de batches e augmentations são idênticos para ambos os cenários na rodada $i$.
     - O conjunto de teste (582 radiografias: 171 sadias e 411 pneumonia) permanece imutável e idêntico em todas as 60 execuções.

2. **Inferência Estatística (Teste $t$ Pareado Bilateral):**
   - Para cada par $i$, calcula-se a diferença pareada:
     $$d_i = X_{1, i} - X_{2, i}$$
   - Estatística do teste:
     $$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$
     onde $\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i$, $s_d$ é o desvio padrão amostral das diferenças e $n = 30$ ($29$ graus de liberdade).
   - Intervalo de Confiança ($95\%$):
     $$\text{IC}_{95\%} = \left[ \bar{d} - t_{0.025, 29} \frac{s_d}{\sqrt{n}}, \quad \bar{d} + t_{0.025, 29} \frac{s_d}{\sqrt{n}} \right]$$
   - **Critério de Decisão:** Rejeita-se $H_0$ se $p\text{-valor} < 0{,}05$.
   - **Resultado Empírico:**
     - F1-Macro: $\bar{d} = +0{,}0997$, $t = 10{,}07$, $p = 5{,}63 \times 10^{-11}$ $\rightarrow$ **Rejeita $H_0$**.
     - Acurácia: $\bar{d} = +0{,}0918$, $t = 8{,}83$, $p = 1{,}03 \times 10^{-09}$ $\rightarrow$ **Rejeita $H_0$**.

3. **Matriz de Confusão e Contagem Amostral no Teste ($N=582$):**
   - **Transfer Learning:** 537 acertos ($92{,}27\%$) e 45 erros ($7{,}73\%$). Sensibilidade de $99{,}27\%$ ($408/411$) e especificidade de $75{,}44\%$ ($129/171$).
   - **Scratch (Warmup):** 500 acertos ($85{,}91\%$) e 82 erros ($14{,}09\%$). Sensibilidade de $98{,}29\%$ ($404/411$) e especificidade de $56{,}14\%$ ($96/171$).
