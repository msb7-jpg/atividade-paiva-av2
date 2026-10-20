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

### [ ] Etapa 2: Template Pandoc IEEE e Base Bibliográfica (.bib)
- [ ] Desenvolver template Pandoc `latex/template-ieee.tex` compatível com `IEEEtran.cls`.
- [ ] Criar base BibTeX formal `latex/referencias.bib` com artigos seminais verificados.
- [ ] Desenvolver script de compilação `scripts/build_paper.sh`.
- [ ] Validar compilação com artigo de teste preliminar gerando PDF em 2 colunas.
- **Commit Previsto:** `build(latex): create pandoc IEEEtran template and verified bibtex base`

### [ ] Etapa 3: Especificação Metodológica do Estudo
- [ ] Formalizar hipóteses científicas ($H_0$ e $H_1$).
- [ ] Definir Técnicas A (ViT/CLIP Features), B (Dropout + L2) e C vs C_alt.
- [ ] Documentar protocolo estatístico pareado ($N=30$) e explicabilidade (XAI).
- **Commit Previsto:** `feat(design): specify formal comparative study protocol and hypotheses`

### [ ] Etapa 4: Implementação do Pipeline Experimental (Python / PyTorch)
- [ ] Implementar `src/experiment.py` com seeds 1001-1030 e pareamento de lotes.
- [ ] Implementar cálculo de Teste t pareado bilateral, IC 95% e F1-macro.
- [ ] Gerar gráficos (Boxplot e XAI) para o artigo.
- [ ] Criar notebook `notebooks/experimento_av2.ipynb` para Colab.
- **Commit Previsto:** `feat(code): implement paired 30-seed simulation pipeline and statistical test`

### [ ] Etapa 5: Redação do Artigo Científico em Padrão IEEE (2 Colunas)
- [ ] Redigir `paper/artigo.md` com pelo menos uma citação seminal por parágrafo.
- [ ] Inserir tabelas formais e figuras de resultados.
- [ ] Compilar PDF final via `scripts/build_paper.sh`.
- **Commit Previsto:** `docs(paper): write complete IEEE double-column paper with citations and compile PDF`

### [ ] Etapa 6: Elaboração dos Slides da Apresentação
- [ ] Redigir roteiro e estrutura dos slides em `slides/apresentacao.md`.
- **Commit Previsto:** `docs(slides): create presentation structure and visual slides outline`
