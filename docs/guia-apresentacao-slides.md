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
| **1** | **Rafaela Vidal** | Contexto Clínico, Desafio e Princípio de Isolamento (*Ceteris Paribus*) | 1 - 2 | `docs/instrucoes-avaliacao.md`<br>`docs/prd.md`<br>`paper/artigo.md` (Seç. I) | Motivação do diagnóstico de pneumonia pediátrica e enquadramento científico $A+B+C$ vs $A+B+C_{\text{alt}}$. |
| **2** | **Miguel Batista** | Base de Dados Real e Técnica A (Data Augmentation Estocástico) | 3 - 4 | `src/repositories/dataset_repository.py`<br>`src/plots/medical_samples.py` | Partição estrita (4.077 treino / 582 teste imutável) e transformações governadas pela semente $s_i$. |
| **3** | **Beatriz Pereira** | Técnica B (ResNet-50) e a Variável Independente (Técnica C vs $C_{\text{alt}}$) | 5 - 6 | `src/services/resnet_builder.py`<br>`src/plots/architecture_diagram.py` | Conexões residuais e a comparação: Transfer Learning ImageNet vs Scratch com He Normal + Cosine Annealing. |
| **4** | **Manuela Cavalcanti** | Protocolo Experimental Pareado ($N=30$) e Orquestração | 7 - 8 | `src/services/experiment_runner.py`<br>`src/domain/models.py`<br>`results/simulacoes_pareadas.json` | Por que $N=30$ (Teorema Central do Limite), pareamento estrito de 60 rodadas e persistência granular. |
| **5** | **Rafael Barros** | Inferência Estatística Formal: Hipóteses e Teste $t$ Pareado | 9 - 10 | `src/services/statistical_service.py`<br>`src/plots/generator.py`<br>`results/boxplot_comparativo.png` | Formulação de $H_0$ e $H_1$, estatística $t = 10{,}07$, $p = 5{,}63 \times 10^{-11}$ e rejeição categórica de $H_0$. |
| **6** | **Ygor Rosa** | Análise Clínica dos Resultados: Matriz de Confusão e Trade-offs | 11 - 12 | `src/domain/models.py`<br>`paper/artigo.md` (Tab. II)<br>`results/simulacoes_pareadas.json` | 537 acertos vs 500 acertos no teste, gravidade clínica do falso negativo ($3$ vs $7$) e colapso de especificidade do Scratch. |
| **7** | **Hugo Alcântara** | Explicabilidade (Grad-CAM), Auditoria LMM e Engenharia da Solução | 13 - 15 | `src/services/gradcam_service.py`<br>`src/services/lmm_insights_service.py`<br>`src/cli.py`<br>`scripts/build_paper.sh` | Heatmaps na `layer4`, laudo automatizado por Gemma 4 (`localhost:8080`), Clean Architecture e CLI do projeto. |

---

## 3. Guias Individuais Detalhados por Integrante

---

### Integrante 1 — Rafaela Vidal
**Tema:** Contexto Clínico, Desafio Diagnóstico e Princípio Científico de Isolamento (*Ceteris Paribus*)

#### 1. Slides sob sua responsabilidade:
- **Slide 1:** Capa do Trabalho, Identificação dos Autores (7 integrantes), Disciplina do Prof. Rodrigo Paiva e CESAR School.
- **Slide 2:** O Desafio Diagnóstico da Pneumonia Pediátrica e a Formulação Científica da 2ª Avaliação.

#### 2. O que você deve falar (Roteiro):
- "Bom dia a todos e ao professor Paiva. O nosso trabalho aborda o diagnóstico assistido por computador de pneumonia a partir de radiografias de tórax pediátricas."
- "A pneumonia é uma das principais causas de mortalidade infantil no mundo, e o diagnóstico depende da identificação de opacidades e infiltrados alveolares no exame radiológico."
- "Para a 2ª Avaliação, nós seguimos estritamente o enunciado formal: comparar a combinação **$A + B + C$** contra **$A + B + C_{\text{alternativa}}$** sob o princípio de **ceteris paribus** ('tudo o mais constante')."
- "Em Machine Learning experimental, se você alterar a arquitetura e o pré-processamento ao mesmo tempo, você gera variáveis de confusão e não consegue provar causalidade. Por isso, mantivemos a Técnica A (Data Augmentation) e a Técnica B (ResNet-50 + Grad-CAM) rigorosamente idênticas, isolando apenas a Técnica C como variável independente."

#### 3. Aonde olhar no código:
- [docs/instrucoes-avaliacao.md](file:///home/miguelsb/workspace/atividade-paiva-av2/docs/instrucoes-avaliacao.md): Enunciado e regras de A/B testing e isolamento de variáveis.
- [docs/prd.md](file:///home/miguelsb/workspace/atividade-paiva-av2/docs/prd.md): Requisitos de produto, problema clínico e premissas.
- [paper/artigo.md](file:///home/miguelsb/workspace/atividade-paiva-av2/paper/artigo.md#L45-L65): Introdução e Seção III-A (Seleção e Isolamento das Técnicas).
- Card Visual Pedagógico: `results/code_cards/card_02_resnet50_ceteris_paribus.png`.

#### 4. O que precisa saber na ponta da língua:
- O que significa *ceteris paribus* e por que ele é obrigatório numa comparação científica.
- Quais são as três técnicas do projeto:
  - **A (Fixa):** Data Augmentation estocástico.
  - **B (Fixa):** Arquitetura ResNet-50 + Grad-CAM + Auditoria LMM multimodal.
  - **C (Variável Independente):** Transfer Learning (ImageNet-1k) vs Treinamento do Zero (Scratch com He Normal + Cosine Annealing).

#### 5. Pergunta capciosa do Professor Paiva & Como responder:
- **Pergunta:** *"Por que vocês não compararam ResNet-50 contra Vision Transformer ou MobileNet?"*
- **Resposta:** *"Professor, se trocássemos a ResNet por ViT ou MobileNet, estaríamos mudando a Técnica B ao mesmo tempo que alteramos o treinamento, violando o isolamento de variável estipulado na avaliação. Para ter causalidade científica pura, a arquitetura deve ser uma constante de controle fixa."*

---

### Integrante 2 — Miguel Batista
**Tema:** Base de Dados Real e Técnica A (Data Augmentation Estocástico)

#### 1. Slides sob sua responsabilidade:
- **Slide 3:** Base de Dados Médica (Chest X-Ray) e Partição Estrita de Avaliação.
- **Slide 4:** Técnica A: Pipeline de Data Augmentation Estocástico e Determinismo.

#### 2. O que você deve falar (Roteiro):
- "Passando para a base de dados: utilizamos a base pública de Kermany et al., amplamente validada na literatura médica, contendo 4.659 radiografias torácicas."
- "Dividimos a base em duas partições bem delimitadas: 4.077 radiografias de treino (1.104 controles sadios e 2.973 pneumonias) e fixamos um conjunto de teste independente com **582 radiografias** (171 sadias e 411 patológicas)."
- "Esse conjunto de teste nunca foi tocado no treino e permaneceu **100% imutável ao longo de todas as 60 rodadas de experimento**, impedindo qualquer vazamento de dados (*data leakage*)."
- "Para a Técnica A, implementamos um pipeline de *Data Augmentation* estocástico no PyTorch: redimensionamento para $224 \times 224$, espelhamento horizontal ($p=0{,}5$), rotação aleatória de $\pm 10^\circ$, perturbação de brilho e contraste (*ColorJitter* $\pm 10\%$) e normalização ImageNet."
- "O ponto crucial de engenharia é que essas transformações são governadas pela semente aleatória $s_i$ de cada rodada, garantindo que o Cenário 1 e o Cenário 2 recebam exatamente o mesmo fluxo de imagens aumentadas na rodada $i$."

#### 3. Aonde olhar no código:
- [src/repositories/dataset_repository.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/repositories/dataset_repository.py#L17-L45): Funções `get_train_transforms()` e `get_test_transforms()`.
- [src/plots/medical_samples.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/plots/medical_samples.py): Gerador visual das amostras clínicas.
- Card Visual Pedagógico: `results/code_cards/card_01_data_augmentation.png` (usar no slide).
- Imagem: `results/amostras_clinicas_xray.png` (usar no slide).

#### 4. O que precisa saber na ponta da língua:
- Por que a rotação é limitada a $\pm 10^\circ$ (em radiografia torácica, um paciente não faz raio-X de cabeça para baixo ou a 90 graus; grandes rotações são anatomicamente irreais).
- Por que normalizamos com a média e desvio padrão do ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- O que é o desbalanceamento da base (mais pneumonias do que normais) e por que isso exige métricas como $F_1\text{-Macro}$ além da acurácia simples.

#### 5. Pergunta capciosa do Professor Paiva & Como responder:
- **Pergunta:** *"O Data Augmentation foi aplicado no conjunto de teste?"*
- **Resposta:** *"Não, professor! No teste aplicamos apenas a transformação determinística (`get_test_transforms`): redimensionamento para $224 \times 224$ e normalização. Augmentation estocástico no teste corromperia a avaliação clínica."*

---

### Integrante 3 — Beatriz Pereira
**Tema:** Técnica B (ResNet-50) e a Variável Independente (Técnica C vs $C_{\text{alt}}$)

#### 1. Slides sob sua responsabilidade:
- **Slide 5:** Técnica B: A Arquitetura Convolucional Residual ResNet-50.
- **Slide 6:** Técnica C (Transfer Learning) versus Técnica $C_{\text{alt}}$ (Scratch com Warmup e Cosine Annealing).

#### 2. O que você deve falar (Roteiro):
- "A nossa Técnica B fixa é a arquitetura ResNet-50 de He et al. Escolhemos uma rede residual profunda porque suas conexões de salto (*skip connections*) somam a entrada $x$ à saída dos blocos residuais $F(x) + x$, mitigando a dissipação de gradientes (*vanishing gradient*) e permitindo o aprendizado de hierarquias visuais complexas."
- "Agora chegamos ao coração da nossa variável independente — a Técnica C:"
  - "**Cenário 1 (Transfer Learning):** Inicializamos a ResNet-50 com os pesos convolucionais pré-treinados na base ImageNet-1k, substituindo apenas a camada densa final para 2 classes (Normal vs Pneumonia). Otimizamos com Adam ($\eta = 10^{-4}$ e *weight decay* $L_2 = 10^{-4}$)."
  - "**Cenário 2 (Scratch / Controle Contrafactual):** Inicializamos todos os pesos convolucionais do zero usando a distribuição He Normal (Kaiming). Para dar uma chance justa ao modelo do zero e não deixá-lo estagnar, implementamos uma política moderna de agendamento: **1 época de Linear Warmup** para estabilizar os gradientes iniciais, seguida de **Cosine Annealing** com $\eta \in [10^{-6}, 3 \times 10^{-4}]$."

#### 3. Aonde olhar no código:
- [src/services/resnet_builder.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/resnet_builder.py#L15-L85):
  - Função `create_resnet50(pretrained=True/False)`.
  - Inicialização `nn.init.kaiming_normal_` nos pesos do Scratch.
  - Classe `LinearWarmupCosineAnnealingLR` com o cálculo matemático do cosseno.
- [src/plots/architecture_diagram.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/plots/architecture_diagram.py): Diagrama comparativo das duas abordagens.

#### 4. O que precisa saber na ponta da língua:
- Como funciona uma *skip connection* na ResNet ($y = \mathcal{F}(x, \{W_i\}) + x$).
- Por que usamos He Normal (Kaiming) para redes com ativação ReLU em vez de Xavier/Glorot (He Normal leva em conta que a ReLU zera metade das ativações, mantendo a variância constante).
- O que é o *Linear Warmup* e por que ele evita que pesos aleatórios recebam gradientes explosivos no início do treino.

#### 5. Pergunta capciosa do Professor Paiva & Como responder:
- **Pergunta:** *"O modelo do zero não perdeu simplesmente porque vocês usaram hiperparâmetros ruins para ele?"*
- **Resposta:** *"Pelo contrário, professor. Nós não usamos SGD puro com taxa fixa. Implementamos Warmup Linear com Cosine Annealing e taxa inicial de $3 \times 10^{-4}$, que é o estado da arte na literatura para treinar CNNs do zero (Goyal et al., Loshchilov et al.). O modelo do zero convergiu muito bem e atingiu 85,9% de acurácia, mas a transferência de aprendizado forneceu priors espaciais de baixo nível que o Scratch não consegue sintetizar com poucos dados."*

---

### Integrante 4 — Manuela Cavalcanti
**Tema:** Protocolo Experimental Pareado ($N=30$) e Orquestração de Execução

#### 1. Slides sob sua responsabilidade:
- **Slide 7:** Por que 30 Simulações? Teorema Central do Limite e Variabilidade Estocástica.
- **Slide 8:** Protocolo Pareado e Orquestrador de Experimentos (`ExperimentRunner`).

#### 2. O que você deve falar (Roteiro):
- "O professor Paiva exigiu a execução de 30 simulações. Por que isso é vital em Inteligência Artificial?"
- "Em redes neurais, rodar um treino uma única vez pode gerar um resultado artificialmente bom ou ruim por mero acaso da semente inicial — o sorteio dos lotes e da ordem dos dados introduz variabilidade estocástica."
- "Pelo **Teorema Central do Limite**, quando a amostragem atinge $N \ge 30$, a distribuição amostral das médias se aproxima de uma normal, viabilizando o uso de testes de hipótese paramétricos com validade estatística."
- "Mais do que 30 rodadas isoladas, nosso protocolo foi **estritamente pareado**: definimos o conjunto de sementes $\mathcal{S} = \{1001, 1002, \dots, 1030\}$, totalizando **60 treinamentos completos**."
- "Na iteração $i$ com a semente $s_i$, o sorteio de batches no PyTorch e os augmentations são espelhados de forma idêntica entre o Transfer Learning e o Scratch. Não houve discrepância de amostragem entre eles."
- "No código, o `ExperimentRunner` gerencia o loop, avalia o teste de 582 imagens a cada rodada e persiste os resultados granulares em `results/runs/run_seed_XXXX.json` e checkpoints dos modelos em `results/checkpoints/`."

#### 3. Aonde olhar no código:
- [src/services/experiment_runner.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/experiment_runner.py#L30-L150): Loop de simulações, `torch.manual_seed(seed)`, persistência granular de runs.
- [src/domain/models.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/domain/models.py#L5-L40): Classes Pydantic `EvaluationMetrics` e `PairedSimulationResult`.
- Arquivo de saída: `results/simulacoes_pareadas.json`.

#### 4. O que precisa saber na ponta da língua:
- O que é uma amostragem pareada (*paired samples*) e por que ela tem mais poder estatístico do que duas amostras independentes (ela cancela a variância causada pela aleatoriedade da semente).
- Quais sementes foram usadas: 1001 a 1030 ($N=30$).
- Quantos treinamentos foram feitos no total: 60 (30 para o Cenário 1 e 30 para o Cenário 2).

#### 5. Pergunta capciosa do Professor Paiva & Como responder:
- **Pergunta:** *"Como vocês garantiram que o treino foi determinístico e reprodutível entre os cenários?"*
- **Resposta:** *"Fixamos `torch.manual_seed(seed)`, `np.random.seed(seed)` e `random.seed(seed)` no início de cada simulação $s_i$ dentro do `ExperimentRunner`. Ambas as rotinas consumiram os mesmos geradores de números pseudo-aleatórios sob a mesma semente."*

---

### Integrante 5 — Rafael Barros
**Tema:** Inferência Estatística Formal: Hipóteses, Teste $t$ de Student Pareado e Decisão

#### 1. Slides sob sua responsabilidade:
- **Slide 9:** Formulação Matemática de $H_0$ e $H_1$, Teste $t$ Pareado Bilateral.
- **Slide 10:** Tabela de Resultados Empíricos e Boxplot Comparativo.

#### 2. O que você deve falar (Roteiro):
- "Esta é a etapa central exigida na avaliação: responder se a diferença entre as técnicas é fruto do acaso ou se reflete uma superioridade real, via inferência estatística formal."
- "Formulamos as hipóteses bilaterais sobre a diferença das médias populacionais de eficácia $\mu_1$ (Transfer Learning) e $\mu_2$ (Scratch):"
  - **Hipótese Nula ($H_0$):** $\mu_1 - \mu_2 = 0$ (não há diferença real; a variação observada é ruído estocástico).
  - **Hipótese Alternativa ($H_1$):** $\mu_1 - \mu_2 \neq 0$ (existe diferença real e estatisticamente significativa).
- "Adotamos o nível de significância clássico $\alpha = 0{,}05$. Calculamos a diferença pareada $d_i = X_{1,i} - X_{2,i}$ para cada par de sementes, e obtivemos a estatística $t$ com 29 graus de liberdade ($df = n-1 = 29$):"
  - **F1-Macro:** Diferença média $\bar{d} = +0{,}0997$ ($+9{,}97\%$), com estatística $t = 10{,}07$ e $p\text{-valor} = 5{,}63 \times 10^{-11}$.
  - **Acurácia:** Diferença média $\bar{d} = +0{,}0918$ ($+9{,}18\%$), com $t = 8{,}83$ e $p\text{-valor} = 1{,}03 \times 10^{-09}$.
  - **Intervalo de Confiança (95%):** $[+0{,}0794, +0{,}1199]$ — note que o zero está totalmente fora do intervalo!
- "Como o $p\text{-valor}$ é de ordem $10^{-11}$, ordens de magnitude inferior a $0{,}05$, **rejeitamos categoricamente a Hipótese Nula ($H_0$) em favor da Hipótese Alternativa ($H_1$)**."
- "O Boxplot (Slide 10) ilustra isso visualmente: a caixa do Transfer Learning é muito mais alta e com dispersão muito mais estreita, provando a robustez e superioridade da transferência de conhecimento."

#### 3. Aonde olhar no código:
- [src/services/statistical_service.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/statistical_service.py#L10-L55): Método `compute_paired_t_test()`, uso de `scipy.stats.ttest_rel()`.
- [src/plots/generator.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/plots/generator.py): Código que renderiza `results/boxplot_comparativo.png`.
- [paper/artigo.md](file:///home/miguelsb/workspace/atividade-paiva-av2/paper/artigo.md#L72-L95): Tabela I do artigo e equações do teste $t$.

#### 4. O que precisa saber na ponta da língua:
- A fórmula do teste $t$ pareado: $t = \frac{\bar{d}}{s_d / \sqrt{n}}$.
- O que significa o $p$-valor ($5{,}63 \times 10^{-11}$): é a probabilidade infinitesimal de observar essa diferença caso a hipótese nula fosse verdadeira.
- Quantos graus de liberdade foram usados: $29$ ($N-1 = 30-1$).
- Por que o teste é bilateral (*two-tailed*): porque a hipótese alternativa testa qualquer diferença ($\neq 0$), sem assumir a priori a direção do ganho.

#### 5. Pergunta capciosa do Professor Paiva & Como responder:
- **Pergunta:** *"Por que vocês usaram o Teste $t$ pareado em vez do Teste $t$ de Student tradicional (independente)?"*
- **Resposta:** *"Porque os treinamentos foram pareados pela mesma semente $s_i$. O teste pareado analisa o vetor das diferenças $d_i = X_{1,i} - X_{2,i}$, o que elimina o efeito da variância entre as sementes e confere muito mais poder estatístico para rejeitar $H_0$."*

---

### Integrante 6 — Ygor Rosa
**Tema:** Análise Clínica dos Resultados: Matriz de Confusão e Impacto Hospitalar

#### 1. Slides sob sua responsabilidade:
- **Slide 11:** Matriz de Confusão e Decomposição Diagnóstica no Teste ($N=582$).
- **Slide 12:** O Impacto Clínico Real: Falsos Negativos vs Falsos Positivos na Prática Médica.

#### 2. O que você deve falar (Roteiro):
- "Passando agora da estatística teórica para a prática médica: o que esses números significam dentro de uma UTI ou emergência pediátrica?"
- "Avaliamos as duas abordagens nas 582 radiografias do conjunto de teste independente (Tabela II do artigo):"
  - **Transfer Learning:** acertou **537 de 582** imagens (**92,27%** de acurácia global), cometendo apenas 45 erros.
  - **Scratch (Warmup):** acertou **500 de 582** imagens (**85,91%** de acurácia), cometendo **82 erros** — quase o dobro de falhas!
- "Agora olhem para a decomposição dos erros clínicos mais perigosos:"
  - **Falsos Negativos (Pneumonia não detectada):**
    - Transfer Learning teve apenas **3 falsos negativos** em 411 pacientes doentes (**sensibilidade de 99,27%**).
    - O Scratch teve **7 falsos negativos** (mais que o dobro).
    - *Em medicina, o falso negativo é catastrófico: é a criança doente que recebe alta sem antibiótico e pode evoluir a óbito em casa.*
  - **Falsos Positivos (Sadio confundido com doente):**
    - Aqui está a maior fragilidade do modelo treinado do zero: ele gerou **75 falsos positivos**, desabando a **especificidade para apenas 56,14%** (contra 75,44% do Transfer Learning).
    - O Scratch tendeu a 'chutar' pneumonia para qualquer anormalidade visual, sobrecarregando o sistema de saúde e prescrevendo antibióticos desnecessários para crianças saudáveis."

#### 3. Aonde olhar no código:
- [src/domain/models.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/domain/models.py#L5-L25): Campos `tp`, `tn`, `fp`, `fn`, `sensitivity`, `specificity`.
- [paper/artigo.md](file:///home/miguelsb/workspace/atividade-paiva-av2/paper/artigo.md#L98-L125): Tabela II (Matriz de Confusão e Contagem Amostral).
- Tabela no terminal: comando `uv run python -m src.cli stats`.

#### 4. O que precisa saber na ponta da língua:
- A diferença entre Sensibilidade (Recall dos doentes: $TP / (TP + FN)$) e Especificidade (Recall dos sadios: $TN / (TN + FP)$).
- O número de imagens de teste: exatamente 582 (171 normais e 411 pneumonia).
- Por que a acurácia isolada é enganosa em problemas médicos desbalanceados e por que a sensibilidade de 99,27% é o grande destaque do Transfer Learning.

#### 5. Pergunta capciosa do Professor Paiva & Como responder:
- **Pergunta:** *"Por que a especificidade de ambos os modelos ficou abaixo de 80%?"*
- **Resposta:** *"Porque a base de Kermany et al. tem quase 3 vezes mais casos de pneumonia do que normais no treino. Isso gera um viés indutivo natural para prever a classe majoritária. Porém, o Transfer Learning conteve muito melhor esse viés, mantendo 75,44% de especificidade, enquanto o Scratch sofreu um colapso para 56,14%, acusando quase metade dos pacientes sadios como pneumônicos."*

---

### Integrante 7 — Hugo Alcântara
**Tema:** Explicabilidade Visual (Grad-CAM), Auditoria Multimodal (LMM) e Engenharia da Solução

#### 1. Slides sob sua responsabilidade:
- **Slide 13:** Explicabilidade Visual com Grad-CAM (`layer4`) e Comparativo Qualitativo.
- **Slide 14:** Auditoria Diagnóstica Multimodal Automatizada via LMM Local (Gemma 4).
- **Slide 15:** Arquitetura de Software da Solução, CLI e Conclusões Finais.

#### 2. O que você deve falar (Roteiro):
- "Para fechar a Técnica B e responder a uma das maiores exigências da IA médica moderna — a interpretabilidade (*XAI*) —, implementamos o algoritmo **Grad-CAM** de Selvaraju et al."
- "Como funciona: interceptamos os mapas de ativação na camada `layer4` da ResNet-50 via *forward hook* e calculamos os gradientes da classe pneumonia via *backward hook*. Fazemos a média global dos gradientes (*global average pooling*) e aplicamos ReLU para destacar apenas os recursos que aumentam a probabilidade de pneumonia."
- "Olhando para a Figura 3 dos nossos resultados:"
  - "O modelo com **Transfer Learning** concentra o mapa térmico (vermelho) com alta precisão no **parênquima pulmonar patológico**, onde estão as consolidações alveolares."
  - "Já o modelo treinado **do zero (Scratch)** dispersa sua atenção para a clavícula, gradil costal e bordas externas da imagem. Ele aprendeu atalhos espúrios (*shortcuts*) em vez de anatomia pulmonar real."
- "Além disso, integramos uma inovação multimodal: conectamos o serviço de auditoria diagnóstica ao **LMM local Gemma 4** rodando em `localhost:8080`. O modelo multimodal recebe a radiografia e o heatmap e emite um laudo radiológico validando que o foco da IA condiz com literatura médica."
- "Em termos de Engenharia de Software, toda a solução foi construída em **Clean Architecture** (Python 3.12, Typer, Rich, Pydantic) com CLI unificada e script de build que compila o artigo IEEE oficial em 4 páginas via Pandoc e pdfLaTeX."

#### 3. Aonde olhar no código:
- [src/services/gradcam_service.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/gradcam_service.py#L12-L65): Classe `GradCAM`, hooks do PyTorch e cálculo do heatmap.
- [src/services/lmm_insights_service.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/services/lmm_insights_service.py#L15-L60): Integração HTTP base64 com a API local em `localhost:8080`.
- [src/plots/xai_visualizer.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/plots/xai_visualizer.py): Geração do painel `results/explicabilidade_xai.png`.
- [src/cli.py](file:///home/miguelsb/workspace/atividade-paiva-av2/src/cli.py) e [scripts/build_paper.sh](file:///home/miguelsb/workspace/atividade-paiva-av2/scripts/build_paper.sh): CLI Typer e pipeline Pandoc/LaTeX.

#### 4. O que precisa saber na ponta da língua:
- Em qual camada da ResNet foi aplicado o Grad-CAM: na última camada convolucional, a **`layer4`** (pois é onde se concentram as representações semânticas espaciais de mais alto nível antes do pooling global).
- Por que se aplica a função ReLU no Grad-CAM: para descartar características que contribuem negativamente para a classe e manter somente o que evidencia a doença.
- Como o LMM foi conectado: servidor de inferência local compatível com a API do OpenAI (`/v1/chat/completions`) recebendo as imagens encodadas em Base64.

#### 5. Pergunta capciosa do Professor Paiva & Como responder:
- **Pergunta:** *"Por que vocês usaram a `layer4` e não a `layer1` ou `layer2` para o Grad-CAM?"*
- **Resposta:** *"As camadas iniciais como `layer1` capturam apenas primitivas visuais de baixo nível — bordas e cantos que não têm significado semântico médico. A `layer4` é a última camada convolucional antes do classificador, preservando resolução espacial rica com abstração semântica suficiente para identificar consolidações pulmonares."*

---

## 4. Guia de Inserção Visual: Qual Imagem e Card de Código vai em Cada Slide

Para que a apresentação tenha alto impacto visual e demonstre o domínio da engenharia e da ciência do projeto, cada slide deve conter sua respectiva evidência gráfica ou card de código gerado em alta resolução:

### Tabela Oficial de Mapeamento Visual (Deck de 9 Slides)

| Slide | Título / Tema | Apresentador(a) | Ativo Visual Recomendado | Caminho do Arquivo | Dica de Diagramação no Slide |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Capa do Trabalho** | Equipe | Logo CESAR School / Ícone de IA Médica | `docs/` | Centralizado ou cabeçalho elegante. |
| **2** | **1. Contexto Clínico: Diagnóstico Pediátrico** | **Rafaela Vidal** | Amostras Clínicas de Radiografias (Sadio vs Pneumonia) | `results/amostras_clinicas_xray.png` | Texto à esquerda; painel vertical com as 2 radiografias à direita para mostrar as opacidades pulmonares. |
| **3** | **2. Isolamento Experimental (*Ceteris Paribus*)** | **Miguel Batista** | **Card 1** (Data Augmentation) **OU** **Card 2** (ResNet Ceteris Paribus) | `results/code_cards/card_01_data_augmentation.png`<br>`results/code_cards/card_02_resnet50_ceteris_paribus.png` | Texto conceitual à esquerda ($A+B+C$ vs $A+B+C_{\text{alt}}$); Card de código à direita comprovando o controle de variáveis. |
| **4** | **3. Arquitetura do Pipeline Experimental** | **Beatriz Pereira** | Diagrama Metodológico Oficial IEEE **e/ou** **Card 3** (DataLoader Pareado) | `results/metodologia_arquitetura.png`<br>`results/code_cards/card_03_paired_dataloader.png` | Diagrama do fluxo à direita, destacando as 30 sementes e o isolamento de geradores estocásticos. |
| **5** | **4. Decisão Baseada em Hipóteses Estatísticas** | **Manuela Cavalcanti** | Fórmulas Matemáticas de $H_0$, $H_1$, Teorema Central do Limite e Pareamento | *(Equações formatadas no slide)* | Fórmulas em destaque central ou lateral com caixas de destaque para $H_0$ e $H_1$. |
| **6** | **5. Resultados da Inferência Estatística ($N=30$)** | **Rafael Barros** | Boxplot Comparativo Oficial (F1 e Acurácia) **e** **Card 4** (Teste $t$) | `results/boxplot_comparativo.png`<br>`results/code_cards/card_04_paired_t_test.png` | Boxplot em tamanho destacado à direita; texto com métricas à esquerda provando $p = 5{,}63 \times 10^{-11}$. |
| **7** | **6. Matriz de Confusão e Diagnóstico de Erros** | **Ygor Rosa** | **Card 5** (Auditoria Clínica da Matriz de Confusão) | `results/code_cards/card_05_confusion_matrix_audit.png` | Card de código à direita com os números 537 vs 500, enfatizando a minimização de Falsos Negativos (3 vs 7). |
| **8** | **7. Explicabilidade Visual (Grad-CAM) e Interpretabilidade** | **Hugo Alcântara** | Painel Grad-CAM Pulmonar **e** **Card 6** (Grad-CAM) / **Card 7** (LMM) | `results/explicabilidade_xai.png`<br>`results/code_cards/card_06_gradcam_pulmonary.png`<br>`results/code_cards/card_07_lmm_insights.png` | Painel XAI com heatmap no centro/direita mostrando o foco no parênquima vs dispersão nas costelas. |
| **9** | **8. Conclusão e Entregáveis do Projeto** | **Hugo / Equipe** | Miniatura do Artigo IEEE de 4 páginas compilado | `paper/pages/page-01.png` | Exibir a capa do artigo oficial à direita, reforçando a conformidade com as exigências da avaliação. |

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
