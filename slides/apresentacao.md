# Roteiro da Apresentação - 2ª Avaliação: Inteligência Artificial
**CESAR School - Outubro/2026**  
**Docente:** Prof. Rodrigo Paiva  
**Discentes:**  
- Miguel Batista (`msb2@cesar.school`)  
- Rafaela Vidal (`rbv2@cesar.school`)  
- Beatriz Pereira (`bcbsp@cesar.school`)  
- Manuela Cavalcanti (`mca2@cesar.school`)  
- Rafael Barros (`ralb@cesar.school`)

---

## Slide 1: Título e Contextualização
- **Título:** Estudo Comparativo de Eficácia em Visão Computacional: Transferência de Representações em Modelos de Fundação vs. Treinamento a Partir do Zero
- **Subtítulo:** Experimento Científico Controlado com 30 Simulações Pareadas
- **Tópicos:**
  - Disciplina: EL1 - Inteligência Artificial
  - Objetivo: Responder formalmente se a transferência de aprendizado é estatisticamente superior à inicialização aleatória ($H_0$ vs. $H_1$).

---

## Slide 2: Motivação e Percurso da Disciplina
- **Do Paradigma Especialista aos Foundation Models:**
  - O percurso anterior: uma arquitetura treinada do zero para cada tarefa isolada.
  - A virada: modelos em escala com auto-supervisão e alinhamento multimodal (CLIP / ViT).
- **Problema de Negócio / Aplicação:**
  - Diagnóstico automatizado de patologias em feijão (*Beans Dataset* - 3 classes: *Angular Leaf Spot*, *Bean Rust* e *Healthy*).
  - Restrição de amostragem no mundo real: apenas centenas de imagens rotuladas disponíveis.

---

## Slide 3: Metodologia e Princípio de Isolamento (*Ceteris Paribus*)
- **Três Técnicas Obrigatórias da Disciplina:**
  - **Técnica A (Fixa):** Extração de representações vetoriais com Vision Transformer / CLIP (`ViT-B/32`, 512 dimensões, patches $14 \times 14$ / $32 \times 32$).
  - **Técnica B (Fixa):** Prevenção rigorosa a overfitting via Dropout ($p=0.20$) + Regularização Ridge ($L_2$, $\lambda = 10^{-4}$) + Otimizador Adam.
  - **Técnica C (Variável Independente):**
    - **Cenário 1 (Solução Principal):** Transfer Learning com pesos pré-treinados multimodais (Linear Probe).
    - **Cenário 2 (Solução Alternativa):** Treinamento com pesos aleatórios estocásticos (*from scratch*).

---

## Slide 4: Fluxograma Arquitetural Comparativo
- **Diagrama em Bloco:**
  - Entrada: $224 \times 224$ RGB $\rightarrow$ Normalização $\rightarrow$ Divisão nos 2 Ramos.
  - Ramo 1: ViT Backbone (Pesos CLIP Pré-treinados).
  - Ramo 2: ViT Backbone (Pesos Aleatórios / Inicialização Xavier).
  - Convergência: Cabeça Comum (Dropout 0.2 $\rightarrow$ Dense 3 Softmax).
  - Treino Pareado: 30 Sementes Idênticas ($s \in [1001, 1030]$) $\rightarrow$ 60 treinamentos no total.

---

## Slide 5: Hipóteses Científicas e Protocolo Estatístico
- **Formulações:**
  - $H_0: \mu_1 - \mu_2 = 0$ (As abordagens têm eficácia idêntica; a diferença observada decorre do acaso).
  - $H_1: \mu_1 - \mu_2 \neq 0$ (Existe diferença estatisticamente significativa ao nível de $\alpha = 0{,}05$).
- **Por que N=30 e Pareamento?**
  - Elimina o ruído estocástico de sorteio de batches e augmentations.
  - Amostra com $N \ge 30$ viabiliza o Teorema Central do Limite e a aplicação do Teste $t$ de Student pareado.

---

## Slide 6: Resultados Quantitativos e Inferência Estatística
- **Tabela Comparativa:**
  - Ganho Médio de F1-Macro: **$+0{,}2861$** ($+28{,}61\%$)
  - Desvio Padrão das Diferenças: $0{,}0501$
  - Estatística $t$: $31{,}27$ ($gl=29$)
  - **Valor-$p$:** $6{,}81 \times 10^{-24}$ (ordens de magnitude abaixo de $0{,}05$)
  - Intervalo de Confiança 95%: $[+0{,}2674, +0{,}3048]$
- **Decisão Formal:** **Rejeição Categórica da Hipótese Nula ($H_0$)** em favor da Hipótese Alternativa ($H_1$).

---

## Slide 7: Análise dos Gráficos (Boxplot Comparativo)
- **Visualização das Distribuições:**
  - Cenário 1 (Transfer): Média de F1-Macro ~ 94,2% com variabilidade mínima (caixa concentrada).
  - Cenário 2 (Aleatório): Média de F1-Macro ~ 64,8% com grande dispersão amostral e instabilidade.
  - Conclusão: Pré-treino traz não apenas maior nota, mas **estabilidade e previsibilidade**.

---

## Slide 8: Explicabilidade Visual (XAI)
- **Mapas de Atenção e Saliência:**
  - O que o modelo com Foundation Model enxerga: foco estrito nas lesões necróticas e manchas de ferrugem da folha.
  - O que o modelo aleatório enxerga: ativações dispersas nas bordas da imagem e ruído de fundo da terra.
  - Corroboração teórica da literatura (Selvaraju et al., 2017).

---

## Slide 9: Conclusão e Entregáveis
- **Síntese dos Entregáveis da 2ª Avaliação:**
  - Código Python modular e reproduzível com CLI direta (`./scripts/run_simulation.sh`).
  - Artigo científico completo no padrão IEEE de 2 colunas compilado via Pandoc + LaTeX (`paper/artigo.pdf`).
  - Base de referências seminais rigorosamente citadas em cada parágrafo.
  - Roteiro e slides para apresentação oral.
