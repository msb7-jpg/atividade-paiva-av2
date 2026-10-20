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

### [ ] Etapa 5: Redação do Artigo Científico em Padrão IEEE (2 Colunas)
- [ ] Redigir `paper/artigo.md` com pelo menos uma citação seminal por parágrafo.
- [ ] Inserir tabelas formais e figuras de resultados.
- [ ] Compilar PDF final via `scripts/build_paper.sh`.
- **Commit Previsto:** `docs(paper): write complete IEEE double-column paper with citations and compile PDF`

### [ ] Etapa 6: Elaboração dos Slides da Apresentação
- [ ] Redigir roteiro e estrutura dos slides em `slides/apresentacao.md`.
- **Commit Previsto:** `docs(slides): create presentation structure and visual slides outline`
