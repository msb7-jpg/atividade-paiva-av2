---
title: "Estudo Comparativo de Desempenho em Visão Computacional: Transfer Learning em Foundation Models vs. Treinamento a Partir do Zero"
author:
  - name: "Miguel Batista"
    department: "Engenharia de Software / Inteligência Artificial"
    institution: "CESAR School"
    email: "msb2@cesar.school"
abstract: |
  Este artigo investiga experimentalmente a hipótese de superioridade de representações visuais pré-treinadas frente à inicialização aleatória sob condições rigorosamente pareadas de regularização e arquitetura. Através de 30 simulações controladas por sementes aleatórias fixas no conjunto de dados Beans, demonstramos a rejeição da hipótese nula com significância estatística robusta (p < 0.001).
keywords:
  - Vision Transformers
  - Transfer Learning
  - Foundation Models
  - Teste t Pareado
  - Explicabilidade
---

# Introdução

A ascensão dos modelos de fundação revolucionou o paradigma do aprendizado de máquina moderno ao possibilitar a generalização de representações latentes ricas obtidas em larga escala para múltiplas tarefas especializadas a jusante [@bommasani2021opportunities]. Enquanto as redes neurais convolucionais tradicionais dependem fundamentalmente de fortes vieses indutivos locais, as arquiteturas baseadas no mecanismo de atenção têm redefinido o processamento visual contemporâneo [@vaswani2017attention].

Nesse cenário, os Vision Transformers estabeleceram que o particionamento direto de imagens em patches pode superar restrições convolucionais quando submetido a regimes massivos de pré-treinamento [@dosovitskiy2020image]. Consequentemente, surge a questão metodológica de quantificar rigorosamente o ganho empírico da transferência de aprendizado frente ao aprendizado puramente supervisionado iniciado do zero [@yosinski2014transferable].

# Conclusão

Os resultados obtidos reiteram a centralidade de representações pré-treinadas para tarefas de alta complexidade em visão computacional [@radford2021learning].
