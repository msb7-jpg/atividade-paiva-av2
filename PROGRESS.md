# Acompanhamento Executivo do Projeto - 2ª Avaliação (Visão Computacional)

## Status Geral: EM ANDAMENTO
- **Data de Início:** 20/10/2026
- **Disciplina:** EL1 – Inteligência Artificial (CESAR School)
- **Professor:** Rodrigo Paiva
- **Último Commit:** Inicializando

---

## Checklists de Progresso por Etapa

### [X] Etapa 1: Saneamento e Organização das Documentações
- [X] Inicializar repositório Git.
- [X] Criar `.gitignore` para arquivos temporários e Zone Identifiers.
- [X] Decodificar e sanear texto original (Windows-1252 / CP850 -> UTF-8).
- [X] Separar documentação em `docs/instrucoes-avaliacao.md` e `docs/exemplo-metodologia-grupo.md`.
- [X] Sanear e padronizar `docs/prd.md`.
- [X] Criar estrutura de diretórios do projeto (`docs/`, `latex/`, `paper/`, `notebooks/`, `src/`, `slides/`, `scripts/`).
- **Commit:** `docs(setup): organize and sanitize project documentation in UTF-8`

### [X] Etapa 2: Template Pandoc IEEE e Base Bibliográfica (.bib)
- [X] Desenvolver template Pandoc `latex/template-ieee.tex` compatível com `IEEEtran.cls` e CSL/Citeproc.
- [X] Criar base BibTeX formal `latex/referencias.bib` com artigos seminais verificados.
- [X] Desenvolver script de compilação `scripts/build_paper.sh`.
- [X] Validar compilação com artigo de teste preliminar gerando PDF em 2 colunas (`paper/artigo.pdf`).
- **Commit:** `build(latex): create pandoc IEEEtran template and verified bibtex base`

### [X] Etapa 3: Especificação Metodológica do Estudo
- [X] Formalizar hipóteses científicas ($H_0$ e $H_1$).
- [X] Definir Técnicas A (ViT/CLIP Features), B (Dropout + L2) e C vs C_alt.
- [X] Documentar protocolo estatístico pareado ($N=30$) e explicabilidade (XAI) em `docs/especificacao-metodologica.md`.
- **Commit:** `feat(design): specify formal comparative study protocol and hypotheses`

### [X] Etapa 4: Implementação do Pipeline Experimental (Python / PyTorch / uv)
- [X] Implementar arquitetura limpa em `src/` (Domínio, Serviços, Visualização, CLI).
- [X] Gerenciar ambiente com `uv` e `pyproject.toml` (Python 3.12+).
- [X] Implementar simulação determinística pareada de 30 seeds (`seeds 1001-1030`).
- [X] Implementar cálculo formal de teste t pareado bilateral, IC 95% e métricas de decisão ($H_0$).
- [X] Criar script executável direto `scripts/run_simulation.sh`.
- [X] Gerar gráficos automáticos de alta resolução em `results/boxplot_comparativo.png`.
- **Commit:** `feat(pipeline): implement clean architecture simulation pipeline and direct runner script`

### [X] Etapa 5: Redação do Artigo Científico em Padrão IEEE (2 Colunas)
- [X] Redigir `paper/artigo.md` com pelo menos uma citação seminal formal em cada parágrafo (NeurIPS, ICML, ICLR, CVPR, ICCV, JMLR).
- [X] Inserir formulação matemática formal de hipóteses ($H_0$, $H_1$).
- [X] Inserir tabela formal do Teste t pareado ($N=30$, gl=29) e figura do Boxplot em alta resolução.
- [X] Compilar com sucesso via `scripts/build_paper.sh`, gerando artigo IEEE em 2 colunas com 3 páginas (`paper/artigo.pdf`).
- **Commit:** `docs(paper): write complete IEEE double-column paper with citations and compile PDF`

### [X] Etapa 6: Elaboração dos Slides da Apresentação
- [X] Redigir roteiro completo e estrutura visual dos slides em `slides/apresentacao.md`.
- [X] Cobrir contextualização, metodologia ceteris paribus, formulação de hipóteses, tabela estatística, boxplots e XAI.
- **Commit:** `docs(slides): create presentation structure and visual slides outline`

### [X] Etapa 7: Transição para Execução Empírica Real com Dados Médicos (Chest X-Ray)
- [X] Documentar plano de transição formal e substituição do Beans Dataset por Radiografias de Tórax (*Chest X-Ray / Pneumonia* - `keremberke/chest-xray-classification`).
- [X] Implementar pipeline de dados com Data Augmentation estocástico em `src/repositories/dataset_repository.py`.
- [X] Implementar CNN ResNet-50 com Transfer Learning (ImageNet) vs Treinamento do Zero (*Scratch*) com Warmup Linear e Cosine Annealing em `src/services/resnet_builder.py`.
- [X] Implementar explicabilidade visual Grad-CAM sobre o parênquima pulmonar em `src/services/gradcam_service.py`.
- [X] Integrar laudo clínico automatizado via LMM multimodal local (`gemma-4-E4B-it-GGUF` em `localhost:8080`) com `thinking: False` em `src/services/lmm_insights_service.py`.
- [X] Implementar gerenciamento dinâmico de VRAM/Compute sob demanda em `src/services/llama_lifecycle.py` liberando 100% da GPU RTX 5060 para os treinos.
- [X] Executar as 30 simulações pareadas reais completas (60 treinamentos no dataset de 4.077 treino e 582 teste).
- [X] Calcular inferência estatística formal ($F_1$-Macro: $\bar{d} = +0{,}0997$, $t = 10{,}07$, $p = 5{,}63 \times 10^{-11}$; Acurácia: $\bar{d} = +0{,}0918$, $t = 8{,}83$, $p = 1{,}03 \times 10^{-09}$), rejeitando categoricamente $H_0$.
- [X] Gerar gráficos em alta resolução: `results/amostras_clinicas_xray.png`, `results/boxplot_comparativo.png`, `results/explicabilidade_xai.png` e `results/laudo_lmm_insights.txt`.
- [X] Atualizar artigo científico `paper/artigo.md` com a nova metodologia clínica, tabela estatística real e referências médicas no BibTeX.
- [X] Validar compilação do PDF IEEE (`paper/artigo.pdf`).
