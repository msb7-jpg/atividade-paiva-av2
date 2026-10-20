---
title: "Estudo Comparativo de Eficácia em Visão Computacional: Transferência de Representações em Modelos de Fundação vs. Inicialização Aleatória"
author:
  - name: "Miguel Batista"
    department: "Engenharia de Software / Inteligência Artificial"
    institution: "CESAR School"
    email: "msb2@cesar.school"
  - name: "Rafaela Vidal"
    department: "Inteligência Artificial"
    institution: "CESAR School"
    email: "rbv2@cesar.school"
  - name: "Beatriz Pereira"
    department: "Inteligência Artificial"
    institution: "CESAR School"
    email: "bcbsp@cesar.school"
  - name: "Manuela Cavalcanti"
    department: "Inteligência Artificial"
    institution: "CESAR School"
    email: "mca2@cesar.school"
  - name: "Rafael Barros"
    department: "Inteligência Artificial"
    institution: "CESAR School"
    email: "ralb@cesar.school"
abstract: |
  A aplicação de modelos de fundação tem transformado o desenvolvimento de soluções em visão computacional através da transferência de aprendizado. Este trabalho investiga experimentalmente a hipótese de superioridade de representações pré-treinadas frente à inicialização aleatória sob condições metodológicas estritamente controladas (princípio ceteris paribus). Utilizando o conjunto de dados Beans, mantivemos invariantes a extração via Vision Transformer (ViT-B/32), mecanismos de regularização com Dropout (0.2) e decaimento de pesos L2, variando exclusivamente a inicialização dos pesos da rede. Conduzimos 30 simulações pareadas indexadas por sementes aleatórias fixas (seeds 1001 a 1030). A aplicação do teste t de Student pareado bilateral revelou um ganho médio de F1-Macro de +0.2861 (IC 95%: [0.2674, 0.3048], p < 1e-20), rejeitando categoricamente a hipótese nula em favor da eficácia comprovada da transferência multimodal.
keywords:
  - Vision Transformers
  - Transfer Learning
  - Foundation Models
  - Teste t Pareado
  - Explicabilidade Visual
---

# I. Introdução

A consolidação dos modelos de fundação inaugurou um novo paradigma no processamento inteligente de imagens, substituindo modelos altamente especializados treinados isoladamente por redes escaladas via auto-supervisão e alinhamento multimodal [@bommasani2021opportunities]. Essa evolução foi viabilizada pela transição das tradicionais operações convolucionais locais para mecanismos globais de auto-atenção, formalizados originalmente na arquitetura Transformer [@vaswani2017attention].

No domínio da percepção visual, os Vision Transformers (ViT) demonstraram que particionar uma imagem em uma sequência unidimensional de *patches* planos permite aprender correlações semânticas de longo alcance de forma superior às convoluções clássicas quando associados a grandes volumes de dados [@dosovitskiy2020image]. Paralelamente, o aprendizado contrastivo de representações visuais com supervisão em linguagem natural viabilizou a emergência de espaços latentes semânticos altamente transferíveis [@radford2021learning].

Contudo, a avaliação empírica do real impacto dessas representações pré-treinadas frente a restrições amostrais severas exige rigor experimental fundamentado no isolamento estrito de variáveis contrafactuais [@yosinski2014transferable]. Este artigo aborda essa lacuna ao conduzir um estudo comparativo controlado com 30 simulações pareadas, testando estatisticamente se o pré-treinamento multimodal de fato supera a inicialização aleatória sob a mesma regularização arquitetural.

# II. Trabalhos Relacionados

A literatura de transferência de aprendizado em visão computacional tem enfatizado historicamente os desafios de viés indutivo e adaptação eficiente de pesos pré-treinados [@chen2020simple]. Para mitigar a dependência excessiva de conjuntos de dados massivos no treinamento de Transformers visuais, técnicas de destilação de conhecimento e aumento sintético de dados provaram-se determinantes para a estabilização de convergência [@touvron2021training].

Adicionalmente, abordagens hierárquicas com janelas de atenção deslocadas demonstraram a versatilidade dos mecanismos de atenção em múltiplas escalas espaciais [@liu2021swin]. No campo do aprendizado auto-supervisionado, modelos mascarados e algoritmos de destilação direta sem anotações humanas consolidaram a capacidade de generalização visual em representações compactas [@he2022masked; @caron2021emerging].

No tocante à prevenção do sobreajuste (*overfitting*) em cabeças de classificação densas, o descarte estocástico de neurônios via Dropout consolidou-se como método padrão da literatura [@srivastava2014dropout]. De maneira complementar, a penalização quadrática dos pesos por meio de decaimento ($L_2$ ou *weight decay*) atua na supressão de co-adaptações espúrias em regimes de baixa amostragem [@krogh1991simple].

# III. Metodologia Proposta

A metodologia segue o preceito do teste controlado (*ablation study*), assegurando a premissa de *ceteris paribus*. O pipeline completo encontra-se ilustrado na Fig. 1 e subdivide-se nas três técnicas curriculares obrigatórias.

![Distribuição comparativa de F1-Macro e Acurácia obtida através das 30 simulações pareadas.](results/boxplot_comparativo.png){width=100%}

## A. Seleção e Isolamento das Técnicas

A **Técnica A (Fixa)** consiste na extração vetorial densa de características com a arquitetura Vision Transformer CLIP (`ViT-B/32`), gerando *embeddings* normalizados de 512 dimensões [@radford2021learning]. As imagens são redimensionadas para $224 \times 224$ pixels e normalizadas de acordo com a distribuição amostral global.

A **Técnica B (Fixa)** atua na regularização da cabeça de decisão compartilhada, incorporando uma camada de *Dropout* com probabilidade $p = 0.20$ seguida de regularização Ridge ($L_2$) com coeficiente de decaimento $\lambda = 10^{-4}$ [@srivastava2014dropout; @krogh1991simple]. O otimizador estocástico adotado é o Adam com taxa de aprendizado inicial $\eta = 10^{-4}$ [@kingma2014adam].

A variável independente é dada pela **Técnica C (Cenário 1)**, que utiliza o extrator de características com pesos consolidados pelo pré-treinamento multimodal (Transfer Learning), em oposição à **Técnica C Alternativa (Cenário 2)**, que emprega inicialização estocástica a partir do zero (*scratch*) sob distribuição Gaussiana truncada.

## B. Formulação das Hipóteses e Protocolo Pareado

Para mitigar a variabilidade estocástica decorrente da inicialização dos lotes e partição de dados, adotamos uma amostragem pareada com $N = 30$ repetições controladas pelo conjunto de sementes $\mathcal{S} = \{1001, 1002, \dots, 1030\}$ [@cohen1988statistical]. Em cada iteração $i$, ambos os cenários são treinados e avaliados sob a mesma semente $s_i$ no conjunto de teste independente composto por 128 instâncias do conjunto de dados Beans.

As hipóteses estatísticas formais são expressas por:
$$\begin{aligned}
H_0 &: \mu_1 - \mu_2 = 0 \\
H_1 &: \mu_1 - \mu_2 \neq 0
\end{aligned}$$
onde $\mu_1$ e $\mu_2$ denotam a média populacional de eficácia (F1-Macro) do Cenário 1 e Cenário 2, respectivamente [@student1908probable]. A rejeição de $H_0$ é estabelecida ao nível bilateral de $\alpha = 0{,}05$.

# IV. Resultados e Análise Estatística

A execução das 60 simulações pareadas confirmou a superioridade expressiva da transferência de representações sobre o treinamento estocástico isolado. A Tabela I sintetiza as estimativas pontuais e intervalares obtidas.

\begin{table}[htbp]
\caption{Resultados do Teste t Pareado Bilateral ($N=30$, gl=29)}
\begin{center}
\begin{tabular}{lcccc}
\toprule
\textbf{Métrica} & \textbf{Diferença Média ($\bar{d}$)} & \textbf{Desvio ($s_d$)} & \textbf{Estatística $t$} & \textbf{Valor-$p$} \\
\midrule
F1-Macro & $+0.2861$ & $0.0501$ & $31.27$ & $6.81 \times 10^{-24}$ \\
Acurácia & $+0.2794$ & $0.0390$ & $39.21$ & $1.12 \times 10^{-26}$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

A diferença média observada no índice F1-Macro foi de $+0{,}2861$, acompanhada de um intervalo de confiança de $95\%$ restrito entre $[0{,}2674, 0{,}3048]$. Dado que o valor-$p$ associado ($6{,}81 \times 10^{-24}$) situa-se ordens de magnitude abaixo do limiar crítico $\alpha = 0{,}05$, rejeita-se categoricamente a hipótese nula ($H_0$).

A dispersão amostral exibida na Fig. 1 atesta a robustez do Cenário 1, cuja variância mostrou-se substancialmente mais contida do que a do Cenário 2, evidenciando que pesos pré-treinados conferem não apenas maior acurácia, mas também resiliência e estabilidade contra variações estocásticas de inicialização [@yosinski2014transferable].

# V. Explicabilidade Visual (XAI)

Para auditar o comportamento qualitativo dos modelos, mapas de saliência baseados em ativação e atenção foram extraídos para amostras representativas das classes *Angular Leaf Spot*, *Bean Rust* e *Healthy* [@selvaraju2017gradcam]. Essa análise evidencia que as atenções no modelo com pesos pré-treinados convergem seletivamente para as áreas necrosadas e pústulas das folhas afetadas.

Em contraste, o modelo com pesos aleatórios dispersa sua atenção pelo contorno foliar e ruídos de iluminação de fundo, explicando a ocorrência frequente de falsos positivos entre patologias adjacentes [@chefer2021transformer]. Essa disparidade qualitativa corrobora a fundamentação teórica de que representações fundamentais codificam conceitos visuais invariantes e semanticamente relevantes.

# VI. Conclusão

Este estudo comparativo comprovou, sob rigorosa inferência estatística pareada com 30 simulações, que a transferência de representações pré-treinadas em modelos de fundação é decisiva para o desempenho e estabilidade em tarefas de classificação com dados reduzidos. As contribuições experimentais e analíticas validam a rejeição de $H_0$ e reforçam a adoção de pipelines estruturados em Vision Transformers para diagnósticos automatizados em visão computacional.
