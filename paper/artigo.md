---
title: "Estudo Comparativo de Otimização em Redes Neurais Convolucionais para Diagnóstico de Pneumonia: Transferência de Aprendizado vs. Treinamento do Zero"
abstract: |
  O diagnóstico assistido por computador em radiografias de tórax demanda modelos de alta acurácia com garantias de interpretabilidade clínica. Este artigo conduz uma investigação empírica rigorosa sobre a eficácia da transferência de aprendizado (*Transfer Learning*) frente ao treinamento a partir do zero (*Scratch*) sob um agendamento com aquecimento linear e decaimento cosseno (*Linear Warmup with Cosine Annealing*), isolando estritamente a variável independente sob o princípio *ceteris paribus*. O pipeline integra Data Augmentation estocástico (Técnica A), arquitetura profunda ResNet-50 com explicabilidade visual Grad-CAM e auditoria clínica via modelo multimodal LMM (Técnica B), variando unicamente o protocolo de inicialização e otimização dos pesos da rede (Técnica C). Foram executadas 30 simulações pareadas indexadas pelas sementes aleatórias 1001 a 1030 (60 treinamentos completos) em 582 radiografias de teste independentes. O teste t de Student pareado bilateral revelou um ganho estatisticamente significativo de F1-Macro de +0.0997 (IC 95%: [0.0794, 0.1199], t = 10.07, p = 5.63e-11) e acurácia de +0.0918 (p = 1.03e-09), refutando a hipótese nula. A explicabilidade visual validada pelo LMM confirmou que a convergência com pesos pré-treinados ancora a atenção no parênquima pulmonar patológico, suprimindo ativações espúrias em estruturas ósseas.
keywords:
  - Radiografia de Tórax
  - Pneumonia
  - ResNet-50
  - Transfer Learning
  - Grad-CAM
  - Teste t Pareado
---

# Introdução

A pneumonia representa uma das principais causas infecciosas de hospitalização e mortalidade global, exigindo triagem diagnóstica precisa através de radiografias de tórax (*Chest X-Ray*) [@kermany2018identifying]. A automatização desse processo por meio de redes neurais convolucionais profundas (CNNs) tem demonstrado alta eficácia na detecção de consolidações e infiltrados alveolares [@rajpurkar2017chexnet]. Contudo, a generalização desses modelos depende criticamente da qualidade das representações visuais aprendidas sob restrições de amostragem clínica e assimetria de classes [@yosinski2014transferable].

No paradigma clássico de aprendizado supervisionado, a transferência de conhecimento oriunda de bases massivas de imagens naturais (como o ImageNet-1k) viabiliza a extração de filtros espaciais fundamentais (bordas, texturas e contornos) capazes de acelerar a convergência e atenuar o sobreajuste (*overfitting*) [@he2016deep]. Em contrapartida, avanços recentes em políticas de agendamento de taxa de aprendizado — notadamente o aquecimento linear seguido de decaimento por cosseno (*Warmup with Cosine Annealing*) — argumentam que o treinamento a partir do zero pode, sob determinadas condições, atingir desempenho competitivo e reduzir vieses específicos de domínios externos [@loshchilov2016sgdr; @goyal2017accurate].

Nesse cenário, torna-se imperativo discernir causalmente se a superioridade frequentemente atribuída ao *Transfer Learning* reflete uma vantagem estatisticamente robusta ou apenas variações estocásticas de inicialização [@he2019rethinking]. Este artigo aborda essa lacuna conduzindo um estudo comparativo controlado com 30 simulações pareadas ($N=30$), submetidas ao Teste $t$ de Student pareado, integrando explicabilidade por Grad-CAM e laudos diagnósticos automatizados por Modelos de Linguagem e Visão Multimodais (LMM).

# Fundamentação Teórica

A literatura médica e de visão computacional consolidou as redes residuais profundas (*Residual Networks* - ResNet) como o padrão ouro para análise de padrões patológicos complexos [@he2016deep]. A incorporação de conexões de salto (*skip connections*) mitiga o problema da dissipação de gradientes, viabilizando o fluxo de sinal através de dezenas de camadas convolucionais hierárquicas [@ronneberger2015u].

Para assegurar a robustez contra variações anatômicas e de scanner radiológico, o aumento estocástico de dados (*Data Augmentation*) atua como regularizador implícito fundamental, simulando variações de orientação e contraste [@shorten2019survey]. De maneira complementar, o método Grad-CAM (*Gradient-weighted Class Activation Mapping*) calcula a contribuição dos gradientes da classe patológica em relação aos mapas de ativação da última camada convolucional, produzindo mapas de calor 2D que indicam a localização espacial dos achados radiológicos [@selvaraju2017gradcam].

Recentemente, a integração de modelos multimodais de fundação (LMMs) tem emergido como mecanismo avançado para auditar a coerência clínica dos mapas de calor de redes profundas, transcrevendo atenções visuais em laudos diagnósticos estruturados em linguagem natural [@radford2021learning; @liu2023visual].

# Metodologia Proposta

A metodologia experimental foi desenhada em estrita conformidade com o princípio de *ceteris paribus*, isolando a estratégia de otimização e inicialização dos pesos como a única variável independente. A Fig. 1 ilustra as instâncias radiológicas avaliadas.

![Painel comparativo de radiografias de tórax do conjunto de teste: (A) Tórax Normal com campos pulmonares translúcidos; (B) Pneumonia com consolidação alveolar patológica.](results/amostras_clinicas_xray.png){width=100%}

## Seleção e Isolamento das Técnicas

A esteira de processamento organiza-se nas três técnicas curriculares:

1. **Técnica A (Fixa / Pré-processamento):** *Data Augmentation* estocástico composto por redimensionamento para $224 \times 224$ pixels, espelhamento horizontal estocástico ($p = 0.50$), rotação aleatória de $\pm 10^\circ$ e perturbação dinâmica de brilho e contraste ($\pm 10\%$). A semente estocástica $s_i$ governa deterministicamente essas transformações em cada rodada.
2. **Técnica B (Fixa / Arquitetura e Explicabilidade):** Classificador convolucional profundo baseado na arquitetura ResNet-50 com *Bottleneck Residual Blocks*, acoplado ao extrator de Grad-CAM sobre a camada `layer4` e ao serviço local de laudo multimodal (LMM) via interface OpenAI-compatible (`localhost:8080`).
3. **Técnica C (Variável Independente):**
   - *Cenário 1 (Transfer Learning):* ResNet-50 inicializada com pesos pré-treinados na base ImageNet-1k, otimizada via Adam ($\eta = 10^{-4}$, decaimento $L_2 = 10^{-4}$).
   - *Cenário 2 (Scratch / Contrafactual):* ResNet-50 inicializada pelo método He Normal [@he2015delving], submetida a 1 época de *Linear Warmup* seguida por decaimento suave de cosseno (*Cosine Annealing*) com taxa de aprendizado $\eta_{\max} = 3 \times 10^{-4}$ e $\eta_{\min} = 10^{-6}$.

## Protocolo Experimental Pareado ($N=30$)

Para garantir inferência estatística sem viés de partição, fixamos o conjunto de teste em 582 radiografias reais (171 sadias e 411 com pneumonia), mantido estritamente imutável durante as 60 execuções. A amostragem pareada percorre as sementes inteiras $\mathcal{S} = \{1001, 1002, \dots, 1030\}$ [@cohen1988statistical].

Em cada simulação pareada $i$, ambos os cenários são submetidos à exata mesma sequência de lotes de treino ($batch = 64$) acelerada nativamente por hardware GPU dedicado. As hipóteses estatísticas bilaterais formulam-se como:
$$\begin{aligned}
H_0 &: \mu_1 - \mu_2 = 0 \\
H_1 &: \mu_1 - \mu_2 \neq 0
\end{aligned}$$
onde $\mu_1$ e $\mu_2$ representam a média populacional de eficácia (F1-Macro e Acurácia) do Cenário 1 e Cenário 2, respectivamente [@student1908probable]. Adota-se o nível crítico bilateral de $\alpha = 0{,}05$.

# Resultados e Análise Estatística

A Tabela I sumariza as estimativas pontuais e intervalares obtidas após a execução completa das 30 simulações pareadas.

\begin{table}[htbp]
\caption{Resultados Empíricos do Teste t Pareado Bilateral ($N=30$, gl=29)}
\label{tab:resultados}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Métrica} & \textbf{Diferença Média ($\bar{d}$)} & \textbf{Desvio ($s_d$)} & \textbf{Estatística $t$} & \textbf{Valor-$p$} \\
\midrule
F1-Macro & $+0.0997$ & $0.0542$ & $10.07$ & $5.63 \times 10^{-11}$ \\
Acurácia & $+0.0918$ & $0.0569$ & $8.83$ & $1.03 \times 10^{-09}$ \\
\bottomrule
\end{tabular}%
}
\end{table}

A análise estatística revelou uma superioridade consistente e altamente expressiva em favor da Transferência de Aprendizado. A diferença média observada no índice F1-Macro foi de $+0{,}0997$, acompanhada por um intervalo de confiança de $95\%$ restrito entre $[0{,}0794, 0{,}1199]$ e estatística $t = 10{,}07$.

Como o valor-$p$ correspondente ($5{,}63 \times 10^{-11}$) situa-se mais de nove ordens de magnitude abaixo de $\alpha = 0{,}05$, rejeita-se categoricamente a hipótese nula ($H_0$) em prol da hipótese alternativa ($H_1$). Resultados análogos foram obtidos para a acurácia global ($\bar{d} = +0{,}0918$, $t = 8{,}83$, $p = 1{,}03 \times 10^{-09}$, $\text{IC}_{95\%}: [0{,}0705, 0{,}1130]$).

![Distribuição comparativa do desempenho empírico ($N=30$ simulações pareadas) entre Transfer Learning e Scratch com Warmup/Cosine Annealing.](results/boxplot_comparativo.png){width=100%}

O boxplot ilustrado na Fig. 2 demonstra a distribuição das métricas. Embora o Cenário 2 tenha demonstrado convergência estável graças ao agendamento de cosseno (atingindo F1-Macro médio superior a $0{,}85$), a Transferência de Aprendizado apresentou dispersão amostral significativamente mais estreita, evidenciando resiliência frente a variações estocásticas de inicialização.

# Explicabilidade Visual (XAI) e Laudo Multimodal

A avaliação qualitativa foi realizada sobre radiografias patológicas através da geração de mapas Grad-CAM e subsequente auditoria diagnóstica por Modelo de Linguagem e Visão Multimodal (LMM) executado localmente. A Fig. 3 apresenta o comparativo de ativação espacial.

![Auditoria qualitativa por Grad-CAM: (A) Radiografia original com pneumonia; (B) Transfer Learning com foco concentrado em parênquima pulmonar; (C) Treinamento do Zero com dispersão em estruturas ósseas claviculares.](results/explicabilidade_xai.png){width=100%}

A auditoria computacional realizada pelo LMM sobre o mapa do Cenário 1 transcreveu formalmente:

> **1. Foco Anatômico:** *A atenção do modelo incide diretamente sobre o parênquima pulmonar direito (campo médio/inferior), sem foco espúrio em bordas ou gradil costal.*  
> **2. Coerência com Pneumonia:** *O padrão destacado coincide com aumento focal de densidade e infiltrado alveolar, achado clássico de pneumonia lobar.*  
> **3. Conclusão Diagnóstica:** *O diagnóstico do classificador é plausível e clinicamente fundamentado em alterações teciduais pulmonares.*

Em contrapartida, conforme demonstrado na Fig. 3(C), o modelo inicializado do zero apresentou gradientes difusos, canalizando parcelas de ativação para as bordas pleurais e articulações claviculares externas, justificando sua menor estabilidade preditiva.

# Conclusão

Sob o isolamento estrito de variáveis e rigorosa validação estocástica com 30 simulações pareadas em dados reais de radiografia torácica, este estudo comprovou que a Transferência de Aprendizado supera estatisticamente o treinamento a partir do zero no diagnóstico de pneumonia ($p < 10^{-10}$). A integração de técnicas de explicabilidade Grad-CAM e laudos emitidos por LMM corroborou a coerência clínica das representações pré-treinadas, validando a rejeição de $H_0$ e consolidando o pipeline proposto como uma solução robusta e interpretável para auxílio radiológico.
