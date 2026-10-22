# Guia Oficial de Apresentação e Divisão de Slides — 2ª Avaliação (IA)

Este documento é o roteiro estratégico e manual de estudo para a equipe na apresentação da **2ª Avaliação de Inteligência Artificial** (CESAR School — Prof. Rodrigo Paiva).

---

## 1. Visão Geral da Apresentação

- **Tempo sugerido:** 15 a 20 minutos totais (~2 a 3 minutos por integrante) + rodada de perguntas da banca.
- **Formato:** Apresentação visual em slides (12 a 15 slides no total) com apoio visual dos gráficos gerados e demonstração rápida de terminal (CLI).
- **Critério Central de Avaliação do Prof. Paiva:**
  1. *Ceteris Paribus:* Demonstrar que apenas uma variável foi alterada entre os cenários.
  2. *Validação Estatística com $N=30$:* Provar rejeição de $H_0$ via Teste $t$ de Student pareado.
  3. *Interpretabilidade e Aplicação Médica:* Justificar os achados com Grad-CAM e auditoria multimodal LMM.
  4. *Domínio da Codebase:* Qualquer membro deve saber exatamente em qual arquivo e função cada técnica foi implementada.

---

## 2. Mapa Geral de Divisão da Apresentação (7 Integrantes)

| # | Integrante | Título da Seção nos Slides | Slide(s) | Arquivos no Código | Foco Principal |
|---|---|---|:---:|---|---|
| **1** | **Rafaela Vidal** | Contexto Clínico, Desafio Diagnóstico e Dataset Chest X-Ray | 1 - 2 | `src/repositories/dataset_repository.py`<br>`docs/prd.md` | Motivação do diagnóstico de pneumonia pediátrica e partição estrita (4.077 treino / 582 teste imutável). |
| **2** | **Miguel Batista** | Princípio de Isolamento (*Ceteris Paribus*) e Técnica A (Data Augmentation) | 3 | `src/repositories/dataset_repository.py`<br>`paper/artigo.md` (Seç. III-A) | Enquadramento científico $A+B+C$ vs $A+B+C_{\text{alt}}$ e transformações estocásticas com restrição anatômica. |
| **3** | **Hugo Alcântara** | Técnica B (ResNet-50), Variável C vs $C_{\text{alt}}$ e Pipeline Pareado | 4 | `src/services/resnet_builder.py`<br>`src/services/experiment_runner.py` | Topologia residual profunda, Transfer vs Scratch com He Normal + Cosine, e isolamento de geradores no DataLoader. |
| **4** | **Rafael Barros** | Hipóteses Matemáticas ($H_0/H_1$) e Resultados do Teste $t$ Pareado ($N=30$) | 5 - 6 | `src/services/statistical_service.py`<br>`results/boxplot_comparativo.png` | Teorema Central do Limite com $N=30$, cálculo de $t = 11{,}23$, $p = 5{,}63 \times 10^{-11}$, Cohen's $d = 2{,}05$ e rejeição de $H_0$. |
| **5** | **Ygor Rosa** | Auditoria Clínica da Matriz de Confusão e Trade-offs Hospitalares | 7 | `src/domain/models.py`<br>`paper/artigo.md` (Tab. II) | 537 acertos vs 500 acertos no teste, minimização de Falsos Negativos (3 vs 7) e colapso de especificidade do Scratch. |
| **6** | **Beatriz Pereira** | Interpretabilidade Visual com Grad-CAM na `layer4` | 8 | `src/services/gradcam_service.py`<br>`results/explicabilidade_xai.png` | Extração de ativações na última camada convolucional, GAP dos gradientes e foco anatômico no parênquima pulmonar. |
| **7** | **Manuela Cavalcanti** | Auditoria Diagnóstica Multimodal com LMM Local e Conclusão | 9 | `src/services/lmm_insights_service.py`<br>`paper/artigo.md` | Integração com Gemma 4 (localhost:8080), emissão de laudo estruturado sem mocks e fechamento dos entregáveis da equipe. |

---

## 3. Guias Individuais Detalhados por Integrante

---

### Integrante 1 — Rafaela Vidal
**Tema:** Contexto Clínico, Desafio Diagnóstico e Dataset Chest X-Ray

#### 1. Slides sob sua responsabilidade:
- **Slide 1:** Capa do Trabalho, Identificação dos Autores (7 integrantes), Disciplina do Prof. Rodrigo Paiva e CESAR School.
- **Slide 2:** O Desafio Clínico da Pneumonia Pediátrica e a Base de Dados Chest X-Ray.

#### 2. O que você deve falar (Roteiro):
- "Bom dia a todos e ao professor Paiva. O nosso trabalho aborda o diagnóstico assistido por computador de pneumonia a partir de radiografias de tórax pediátricas."
- "A pneumonia é uma das principais causas de mortalidade infantil no mundo, e o diagnóstico depende da identificação de opacidades e infiltrados alveolares no exame radiológico."
- "Utilizamos a base pública de Kermany et al., contendo 4.077 radiografias de treino (1.104 controles sadios e 2.973 pneumonias) e fixamos um conjunto de teste independente com **582 radiografias** (171 sadias e 411 patológicas)."
- "Esse conjunto de teste nunca foi tocado no treino e permaneceu **100% imutável ao longo de todas as 60 rodadas de experimento**, impedindo qualquer vazamento de dados (*data leakage*)."
- "Na medicina pediátrica, o Falso Negativo é o erro mais grave: uma criança doente recebe alta indevida sem antibiótico."

#### 3. Aonde olhar no código:
- [src/repositories/dataset_repository.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/repositories/dataset_repository.py): Classe `DatasetRepository`, partições de dados.
- [docs/prd.md](file:///home/miguelsb/workspace/atividade-paiva-av2/docs/prd.md): Requisitos de produto, problema clínico e premissas.
- Card Visual Pedagógico: `results/code_cards/card_01_rafaela_vidal_dataset_chest_xray.png`.
- Imagem Clínica: `results/amostras_clinicas_xray.png` (usar no slide).

---

### Integrante 2 — Miguel Batista
**Tema:** Princípio de Isolamento (*Ceteris Paribus*) e Técnica A (Data Augmentation Estocástico)

#### 1. Slides sob sua responsabilidade:
- **Slide 3:** Princípio Científico de Isolamento (*Ceteris Paribus*) e Técnica A (*Data Augmentation*).

#### 2. O que você deve falar (Roteiro):
- "Para a 2ª Avaliação, nós seguimos estritamente o enunciado formal: comparar a combinação **$A + B + C$** contra **$A + B + C_{\text{alternativa}}$** sob o princípio de **ceteris paribus** ('tudo o mais constante')."
- "Em Machine Learning experimental, se você alterar a arquitetura e o pré-processamento ao mesmo tempo, você gera variáveis de confusão e não consegue provar causalidade. Por isso, mantivemos a Técnica A (Data Augmentation) e a Técnica B (ResNet-50 + Grad-CAM) rigorosamente idênticas, isolando apenas a Técnica C como variável independente."
- "Para a Técnica A, implementamos um pipeline de *Data Augmentation* estocástico no PyTorch: redimensionamento para $224 \times 224$, espelhamento horizontal ($p=0{,}5$), rotação aleatória de $\pm 10^\circ$, perturbação de brilho e contraste (*ColorJitter* $\pm 10\%$) e normalização ImageNet."
- "A rotação é estritamente limitada a $\pm 10^\circ$ e o espelhamento é apenas horizontal: em radiografia torácica, um paciente não faz raio-X de cabeça para baixo; grandes rotações são anatomicamente irreais."

#### 3. Aonde olhar no código:
- [src/repositories/dataset_repository.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/repositories/dataset_repository.py#L16-L35): Funções `get_train_transforms()` e `get_test_transforms()`.
- [paper/artigo.md](file:///home/miguelsb/workspace/atividade-paiva-av2/paper/artigo.md#L45-L65): Introdução e Seção III-A (Seleção e Isolamento das Técnicas).
- Card Visual Pedagógico: `results/code_cards/card_02_miguel_batista_data_augmentation.png`.

---

### Integrante 3 — Hugo Alcântara
**Tema:** Técnica B (ResNet-50), Variável Independente C vs $C_{\text{alt}}$ e Protocolo Pareado

#### 1. Slides sob sua responsabilidade:
- **Slide 4:** Técnica B (ResNet-50), Comparação C vs $C_{\text{alt}}$ e Pareamento de DataLoader.

#### 2. O que você deve falar (Roteiro):
- "A nossa Técnica B fixa é a arquitetura residual ResNet-50 de He et al. Suas conexões de salto (*skip connections*) somam a entrada $x$ à saída dos blocos residuais $F(x) + x$, mitigando a dissipação de gradientes."
- "A variável independente pura (Técnica C vs $C_{\text{alt}}$):"
  - "**Cenário 1 (Transfer Learning):** Inicializamos a ResNet-50 com os pesos convolucionais pré-treinados no ImageNet-1k, substituindo a camada densa para 2 classes. Otimizamos com Adam ($\eta = 10^{-4}$, *weight decay* $L_2 = 10^{-4}$)."
  - "**Cenário 2 (Scratch / Controle Contrafactual):** Inicializamos todos os pesos do zero com He Normal (Kaiming), aplicando 1 época de Linear Warmup para estabilização de gradientes, seguida de Cosine Annealing até $\eta = 10^{-6}$."
- "Além disso, implementamos o **pareamento estrito de DataLoader**: na rodada sob a semente $s_i$, geradores estocásticos isolados garantem que ambos os cenários recebam rigorosamente a mesma ordem de lotes e transformações."

#### 3. Aonde olhar no código:
- [src/services/resnet_builder.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/resnet_builder.py): `create_resnet50()`, inicialização `kaiming_normal_` e classe `LinearWarmupCosineAnnealingLR`.
- [src/services/experiment_runner.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/experiment_runner.py): DataLoader isolado com gerador dedicado por semente.
- Card Visual Pedagógico: `results/code_cards/card_03_hugo_alcantara_resnet50_pipeline.png`.
- Imagem: `results/metodologia_arquitetura.png`.

---

### Integrante 4 — Rafael Barros
**Tema:** Hipóteses Matemáticas ($H_0/H_1$) e Resultados Empíricos do Teste $t$ Pareado ($N=30$)

#### 1. Slides sob sua responsabilidade:
- **Slide 5:** Formulação Matemática de $H_0$ e $H_1$, Teorema Central do Limite e Teste $t$ Pareado Bilateral.
- **Slide 6:** Síntese Numérica dos Resultados Empíricos e Boxplot Comparativo Oficial.

#### 2. O que você deve falar (Roteiro):
- "Avaliamos formalmente se a diferença observada é ruído estocástico ou superioridade estatística real."
- "Formulamos as hipóteses bilaterais sobre a diferença das médias populacionais $\mu_1$ (Transfer) e $\mu_2$ (Scratch):"
  - **$H_0$ (Hipótese Nula):** $\mu_1 - \mu_2 = 0$ (eficácia diagnóstica idêntica; qualquer variação é ruído).
  - **$H_1$ (Hipótese Alternativa):** $\mu_1 - \mu_2 \neq 0$ (diferença real e estatisticamente significativa com $\alpha = 0{,}05$).
- "Pelo **Teorema Central do Limite**, com $N = 30$ pares ($df = 29$), a distribuição das diferenças pareadas converge para a normalidade, permitindo o Teste $t$ com alta potência estatística."
- "Resultados reais da GPU:"
  - **F1-Score Macro:** Transfer $0{,}9555 \pm 0{,}0094$ vs Scratch $0{,}8540 \pm 0{,}0526$. Diferença média $\bar{d} = +0{,}1015$ ($+10{,}15\%$), $t = 11{,}23$, $p = 5{,}63 \times 10^{-11}$, Cohen's $d = 2{,}05$ (Efeito Enorme!).
  - **Acurácia Global:** Transfer $0{,}9640 \pm 0{,}0074$ vs Scratch $0{,}8709 \pm 0{,}0464$. $\bar{d} = +0{,}0931$, $t = 11{,}49$, $p = 3{,}18 \times 10^{-11}$.
- "**Rejeitamos categoricamente a Hipótese Nula ($H_0$) em prol da Hipótese Alternativa ($H_1$)**."

#### 3. Aonde olhar no código:
- [src/services/statistical_service.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/statistical_service.py): `compute_paired_t_test()`.
- [src/plots/generator.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/plots/generator.py): Boxplot comparativo.
- Card Visual Pedagógico: `results/code_cards/card_04_rafael_barros_paired_t_test.png`.
- Imagem: `results/boxplot_comparativo.png`.

---

### Integrante 5 — Ygor Rosa
**Tema:** Auditoria Clínica da Matriz de Confusão e Trade-offs Hospitalares

#### 1. Slides sob sua responsabilidade:
- **Slide 7:** Matriz de Confusão, Decomposição de Erros e Impacto Clínico no Teste ($N=582$).

#### 2. O que você deve falar (Roteiro):
- "Passando agora da estatística teórica para a prática médica: o que esses números significam dentro de uma emergência pediátrica?"
- "Avaliamos as duas abordagens nas 582 radiografias do conjunto de teste independente (Tabela II do artigo):"
  - **Transfer Learning (Cenário 1):** acertou **537 de 582** imagens (**92,27%** de acurácia global), cometendo apenas 45 erros.
  - **Scratch (Cenário 2):** acertou **500 de 582** imagens (**85,91%** de acurácia), cometendo **82 erros** — quase o dobro de falhas!
- "Agora olhem para a decomposição dos erros clínicos mais perigosos:"
  - **Falsos Negativos (Pneumonia não detectada):**
    - Transfer Learning teve apenas **3 falsos negativos** em 411 pacientes doentes (**sensibilidade de 99,27%**).
    - O Scratch teve **7 falsos negativos** (mais que o dobro).
    - *Em medicina, o falso negativo é catastrófico: é a criança doente que recebe alta sem antibiótico e pode evoluir a óbito em casa.*
  - **Falsos Positivos (Sadio confundido com doente):**
    - O Scratch gerou **75 falsos positivos**, desabando a **especificidade para apenas 56,14%** (contra 75,44% do Transfer Learning). O Scratch tende a acusar qualquer anormalidade visual como pneumonia, gerando sobrecarga hospitalar."

#### 3. Aonde olhar no código:
- [src/domain/models.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/domain/models.py): Campos `tp`, `tn`, `fp`, `fn`, `sensitivity`, `specificity`.
- [paper/artigo.md](file:///home/miguelsb/workspace/atividade-paiva-av2/paper/artigo.md#L98-L125): Tabela II do artigo.
- Card Visual Pedagógico: `results/code_cards/card_05_ygor_rosa_confusion_matrix_audit.png`.

---

### Integrante 6 — Beatriz Pereira
**Tema:** Interpretabilidade Visual com Grad-CAM na `layer4`

#### 1. Slides sob sua responsabilidade:
- **Slide 8:** Explicabilidade Visual com Grad-CAM na `layer4` e Foco no Parênquima Pulmonar.

#### 2. O que você deve falar (Roteiro):
- "Para atender à exigência de explicabilidade e interpretabilidade médica (XAI), implementamos o algoritmo **Grad-CAM** de Selvaraju et al."
- "Como funciona: interceptamos os mapas de ativação na última camada convolucional da ResNet-50, a **`layer4`**, via forward hook, e extraímos os gradientes da classe prevista via backward hook."
- "Fazemos a média espacial dos gradientes (*global average pooling*) gerando os pesos $\alpha_k$, combinamos linearmente os canais e aplicamos retificação **ReLU** para destacar apenas ativações que aumentam a probabilidade de pneumonia."
- "Olhando para o painel Grad-CAM dos nossos resultados:"
  - "O modelo com **Transfer Learning** concentra o mapa térmico (vermelho) com alta precisão no **parênquima pulmonar patológico**, onde estão as consolidações alveolares."
  - "Já o modelo treinado **do zero (Scratch)** dispersa sua atenção para clavículas, costelas e bordas externas da radiografia. Ele aprendeu atalhos espúrios (*shortcuts*) em vez de anatomia pulmonar real."

#### 3. Aonde olhar no código:
- [src/services/gradcam_service.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/gradcam_service.py): Classe `GradCAM`, hooks do PyTorch e normalização do mapa de calor.
- [src/plots/xai_visualizer.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/plots/xai_visualizer.py): Geração do painel `results/explicabilidade_xai.png`.
- Card Visual Pedagógico: `results/code_cards/card_06_beatriz_pereira_gradcam_pulmonary.png`.
- Imagem: `results/explicabilidade_xai.png`.

---

### Integrante 7 — Manuela Cavalcanti
**Tema:** Auditoria Diagnóstica Multimodal com LMM Local e Conclusão dos Entregáveis

#### 1. Slides sob sua responsabilidade:
- **Slide 9:** Auditoria Multimodal Automatizada via LMM Local (Gemma 4) e Conclusão dos Entregáveis.

#### 2. O que você deve falar (Roteiro):
- "Para completar a interpretabilidade, integramos uma inovação multimodal: conectamos o serviço de auditoria diagnóstica ao **LMM multimodal local Gemma 4 (4B)** rodando em `localhost:8080` via llama-server."
- "O modelo de visão e linguagem recebe a radiografia original em conjunto com o mapa de calor Grad-CAM e emite um **laudo estruturado** avaliando a coerência anatômica da predição."
- "Um ponto crucial de engenharia: implementamos tratamento estrito de fallback. Se o LMM estiver offline, o sistema não gera laudos falsos simulados: ele avisa explicitamente `[AVISO: LAUDO NÃO GERADO]`, preservando a integridade dos dados."
- "Para concluir, a equipe entrega com êxito os 4 pilares da avaliação:"
  1. Artigo Científico no padrão IEEE oficial de 4 páginas em 2 colunas;
  2. Codebase modular em Clean Architecture com CLI completa;
  3. Galeria Clínica Multimodal (TP, TN, FP, FN) com laudos auditados e heatmaps Grad-CAM em 300 DPI;
  4. Repositório público com rastreabilidade integral de 60 treinamentos reproduzíveis."

#### 3. Aonde olhar no código:
- [src/services/lmm_insights_service.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/lmm_insights_service.py): Chamada HTTP com imagens Base64 e tratamento `LMMUnavailableError`.
- [paper/artigo.md](file:///home/miguelsb/workspace/atividade-paiva-av2/paper/artigo.md): Conclusão e Tabela de entregáveis.
- Card Visual Pedagógico: `results/code_cards/card_07_manuela_cavalcanti_lmm_insights.png`.
- Imagem: `paper/pages/page-01.png`.

---

## 4. Guia de Inserção Visual: Qual Imagem e Card de Código vai em Cada Slide

Para que a apresentação tenha alto impacto visual e demonstre o domínio da engenharia e da ciência do projeto, cada slide deve conter sua respectiva evidência gráfica ou card de código gerado em alta resolução:

### Tabela Oficial de Mapeamento Visual (Deck de 9 Slides)

| Slide | Título / Tema | Apresentador(a) | Ativo Visual Recomendado | Caminho do Arquivo | Dica de Diagramação no Slide |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Capa do Trabalho** | Equipe | Logo CESAR School / Ícone de IA Médica | `docs/` | Centralizado ou cabeçalho elegante com os 7 autores. |
| **2** | **1. Contexto Clínico: Diagnóstico Pediátrico** | **Rafaela Vidal** | Amostras Clínicas **e** **Card 1** (Dataset) | `results/amostras_clinicas_xray.png`<br>`results/code_cards/card_01_rafaela_vidal_dataset_chest_xray.png` | Texto à esquerda; painel vertical com as 2 radiografias e card do dataset à direita. |
| **3** | **2. Isolamento Experimental (*Ceteris Paribus*)** | **Miguel Batista** | **Card 2** (Data Augmentation com Restrição Anatômica) | `results/code_cards/card_02_miguel_batista_data_augmentation.png` | Texto conceitual $A+B+C$ vs $A+B+C_{\text{alt}}$ à esquerda; Card de código à direita comprovando o controle de variáveis. |
| **4** | **3. Arquitetura do Pipeline Experimental** | **Hugo Alcântara** | Diagrama Metodológico Oficial IEEE **e** **Card 3** (ResNet-50 & Pipeline) | `results/metodologia_arquitetura.png`<br>`results/code_cards/card_03_hugo_alcantara_resnet50_pipeline.png` | Diagrama do fluxo à direita, destacando as 30 sementes e o isolamento de geradores estocásticos. |
| **5** | **4. Decisão Baseada em Hipóteses Estatísticas** | **Rafael Barros** | Fórmulas Matemáticas de $H_0$, $H_1$, Teorema Central do Limite e Pareamento | *(Equações formatadas no slide)* | Fórmulas em destaque central ou lateral com caixas de destaque para $H_0$ e $H_1$. |
| **6** | **5. Resultados da Inferência Estatística ($N=30$)** | **Rafael Barros** | Boxplot Comparativo Oficial **e** **Card 4** (Teste $t$ Pareado) | `results/boxplot_comparativo.png`<br>`results/code_cards/card_04_rafael_barros_paired_t_test.png` | Boxplot em tamanho destacado à direita; texto com métricas à esquerda provando $p = 5{,}63 \times 10^{-11}$. |
| **7** | **6. Matriz de Confusão e Diagnóstico de Erros** | **Ygor Rosa** | **Card 5** (Auditoria Clínica da Matriz de Confusão) | `results/code_cards/card_05_ygor_rosa_confusion_matrix_audit.png` | Card de código à direita com os números 537 vs 500, enfatizando a minimização de Falsos Negativos (3 vs 7). |
| **8** | **7. Explicabilidade Visual: Grad-CAM na `layer4`** | **Beatriz Pereira** | Painel Grad-CAM Pulmonar **e** **Card 6** (Grad-CAM `layer4`) | `results/explicabilidade_xai.png`<br>`results/code_cards/card_06_beatriz_pereira_gradcam_pulmonary.png` | Painel XAI com heatmap no centro/direita mostrando o foco no parênquima vs dispersão nas costelas. |
| **9** | **8. Auditoria Multimodal (LMM Local) e Conclusão** | **Manuela Cavalcanti** / **Equipe** | **Card 7** (LMM Local) **e** Miniatura do Artigo IEEE | `results/code_cards/card_07_manuela_cavalcanti_lmm_insights.png`<br>`paper/pages/page-01.png` | Card de código do LMM à esquerda e miniatura do artigo oficial de 4 páginas à direita. |

---

## 5. Guia de Demonstração ao Vivo no Terminal (CLI)

Durante a apresentação ou perguntas, qualquer integrante pode demonstrar os comandos executáveis do projeto no terminal:

```bash
# 1. Verificar inventário das 30 simulações e integridade dos checkpoints
uv run python -m src.cli status

# 2. Exibir a tabela formal com o Teste t Pareado, p-valor e Matriz de Confusão
uv run python -m src.cli stats

# 3. Gerar a visualização do Grad-CAM com auditoria LMM
uv run python -m src.cli xai

# 4. Gerar a galeria clínica completa de explicabilidade
uv run python -m src.cli gallery

# 5. Compilar o artigo científico IEEE de 4 páginas
./scripts/build_paper.sh
```

---

## 6. Dicionário Técnico Rápido para Ninguém se Confundir

| Termo | Definição Rápida para Usar na Apresentação |
|---|---|
| **Ceteris Paribus** | Expressão em latim que significa "tudo o mais constante". Princípio científico que garante que apenas a variável independente muda, isolando causalidade. |
| **Hipótese Nula ($H_0$)** | Afirmação de que não existe efeito ou diferença real entre as abordagens ($\mu_1 - \mu_2 = 0$). Foi categoricamente rejeitada. |
| **Hipótese Alternativa ($H_1$)** | Afirmação de que existe diferença real e estatisticamente significativa entre as abordagens ($\mu_1 - \mu_2 \neq 0$). Foi aceita com $p < 10^{-10}$. |
| **Graus de Liberdade ($df = 29$)** | Em um teste pareado com $N=30$, o número de graus de liberdade é sempre $N - 1 = 29$. |
| **Grad-CAM** | Técnica de interpretabilidade visual que calcula a média ponderada dos mapas de ativação pelos gradientes da classe alvo na última camada convolucional. |
| **Warmup + Cosine Annealing** | Política de taxa de aprendizado que sobe linearmente no início (evitando divergência em pesos aleatórios) e decai suavemente como um cosseno até um piso mínimo. |
| **F1-Macro** | Média harmônica balanceada entre precisão e recall calculada individualmente para cada classe e depois mediada, ideal para dados médicos desbalanceados. |
