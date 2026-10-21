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
- **Título:** Estudo Comparativo de Eficácia em Visão Computacional: Transferência de Representações vs. Treinamento do Zero no Diagnóstico de Pneumonia
- **Subtítulo:** Investigação Empírica Controlada com 30 Simulações Pareadas e Auditoria Clínica Multimodal
- **Tópicos:**
  - Disciplina: EL1 - Inteligência Artificial
  - Objetivo: Responder formalmente se o Transfer Learning é estatisticamente superior ao treinamento do zero com agendamento avançado em dados médicos ($H_0$ vs. $H_1$).

---

## Slide 2: Motivação e Desafio Clínico
- **Diagnóstico Assistido por Computador em Radiografias de Tórax:**
  - Pneumonia como causa primária global de internação e óbito respiratório.
  - Imagens de raio-X possuem sutilezas patológicas (consolidações e infiltrados alveolares difusos).
- **Amostragem Clínica Real:**
  - Dataset *Chest X-Ray* (Kermany et al.): 4.077 radiografias de treino e 582 de teste independente intocável.
  - Forte assimetria de classes e desafios de generalização sob variações estocásticas.

---

## Slide 3: Metodologia e Princípio de Isolamento (*Ceteris Paribus*)
- **Três Técnicas Curriculares Obrigatórias:**
  - **Técnica A (Fixa):** *Data Augmentation* estocástico ($224 \times 224$ px, Flip Horizontal $p=0.5$, Rotação $\pm 10^\circ$, Jitter fotométrico de brilho/contraste). Semente $s_i$ fixa por rodada.
  - **Técnica B (Fixa):** Arquitetura ResNet-50 profunda acoplada a explicabilidade Grad-CAM (`layer4`) e laudo multimodal via LMM local (Gemma 4 em `localhost:8080`).
  - **Técnica C (Variável Independente):**
    - **Cenário 1 (Proposto):** Transfer Learning com pesos pré-treinados ImageNet-1k, otimizador Adam ($\eta = 10^{-4}$, decaimento $L_2 = 10^{-4}$).
    - **Cenário 2 (Contrafactual):** Treinamento do zero com He Normal, submetido a 1 época de *Linear Warmup* e *Cosine Annealing* ($\eta \in [10^{-6}, 3 \times 10^{-4}]$).

---

## Slide 4: Fluxograma Arquitetural Comparativo
- **Diagrama em Bloco Oficial IEEE:**
  - Entrada: $224 \times 224$ RGB $\rightarrow$ Data Augmentation Estocástico sob semente $s_i$.
  - Ramo 1: ResNet-50 com Pesos Pré-treinados ImageNet.
  - Ramo 2: ResNet-50 com Inicialização He Normal do zero + Warmup/Cosine.
  - Avaliação: Teste determinístico sobre as exatas 582 radiografias independentes.
  - Auditoria: Grad-CAM na última camada convolucional + Parecer Clínico automatizado pelo LMM.

---

## Slide 5: Hipóteses Científicas e Protocolo Pareado ($N=30$)
- **Formulações:**
  - $H_0: \mu_1 - \mu_2 = 0$ (As abordagens têm eficácia idêntica; a diferença observada decorre do acaso).
  - $H_1: \mu_1 - \mu_2 \neq 0$ (Existe diferença estatisticamente significativa ao nível de $\alpha = 0{,}05$).
- **Por que N=30 e Pareamento?**
  - Conjunto fixo de sementes $\mathcal{S} = \{1001, \dots, 1030\}$ totalizando 60 treinamentos completos.
  - Elimina variações estocásticas de lote e amostragem.
  - Viabiliza a aplicação do Teste $t$ de Student pareado com 29 graus de liberdade.

---

## Slide 6: Resultados Quantitativos e Inferência Estatística
- **Resultados do Teste t Pareado Bilateral ($N=30$, gl=29):**
  - **Diferença Média F1-Macro:** $+0{,}0997$ ($s_d = 0{,}0542$)
  - **Estatística $t$ (F1):** $10{,}07$ | **Valor-$p$:** $5{,}63 \times 10^{-11}$
  - **Intervalo de Confiança 95% (F1):** $[+0{,}0794, +0{,}1199]$
  - **Diferença Média de Acurácia:** $+0{,}0918$ ($t = 8{,}83$, $p = 1{,}03 \times 10^{-09}$)
- **Decisão Formal:** **Rejeição Categórica de $H_0$** em prol de $H_1$.

---

## Slide 7: Decomposição Amostral e Matriz de Confusão
- **Desempenho no Conjunto de Teste ($N=582$):**
  - **Transfer Learning:**
    - Acertos: **537 de 582 ($92{,}27\%$)** | Erros: **45 ($7{,}73\%$)**
    - Sensibilidade (Pneumonia): **$99{,}27\%$** ($408$ de $411$ detectadas, apenas 3 falsos negativos).
    - Especificidade (Sadio): **$75{,}44\%$** ($129$ de $171$).
  - **Scratch com Warmup/Cosine:**
    - Acertos: **500 de 582 ($85{,}91\%$)** | Erros: **82 ($14{,}09\%$)** — *quase o dobro de erros!*
    - Sensibilidade: **$98{,}29\%$** ($404/411$).
    - Especificidade: **$56{,}14\%$** ($96/171$, com 75 falsos positivos).
- **Impacto Clínico:** Transfer Learning é vital para reduzir falsos negativos críticos e evitar sobrecarga de triagem médica.

---

## Slide 8: Explicabilidade Visual (Grad-CAM) e Laudo Multimodal (LMM)
- **Comparativo de Mapas de Calor:**
  - **Transfer Learning:** Atenção concentrada no parênquima pulmonar direito/inferior (áreas de consolidação alveolar).
  - **Scratch (Aleatório):** Gradientes dispersos nas bordas claviculares e estruturas ósseas externas.
- **Auditoria Automatizada via Gemma 4 (LMM em localhost:8080):**
  - Confirmação de foco anatômico compatível com achados radiológicos.
  - Galeria completa cobrindo Verdadeiros Positivos, Sadios e casos de erro em `results/gallery/`.

---

## Slide 9: Conclusão e Entregáveis
- **Síntese dos Entregáveis da 2ª Avaliação:**
  - Codebase em Clean Architecture com CLI completa (`status`, `run`, `xai`, `gallery`, `diagram`).
  - Persistência granular e tolerante a falhas (`results/runs/` e `results/checkpoints/`).
  - Artigo científico no padrão IEEE de 2 colunas com 4 páginas (`paper/artigo.pdf`).
  - Galeria clínica com mapas de calor e laudos estruturados (`results/gallery/`).
  - Todas as hipóteses e decisões estatísticas rigorosamente validadas.
