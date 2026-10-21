# AGENTS.md — Guia de Arquitetura e Engenharia do Repositório

Bem-vindo ao repositório do **Estudo Comparativo Científico de Visão Computacional (2ª Avaliação)**.  
Este documento é a referência central e mandatória para agentes de Inteligência Artificial e desenvolvedores humanos que operem, mantenham ou expandam esta codebase.

---

## 1. Contexto Acadêmico e Institucional

- **Instituição:** CESAR School — Recife, PE, Brasil
- **Curso / Período:** Engenharia de Software / Inteligência Artificial (Outubro/2026)
- **Disciplina:** EL1 – Inteligência Artificial
- **Docente:** Prof. Rodrigo Paiva (`rfapo@cesar.school`)
- **Equipe de Autores:**
  - Miguel Batista (`msb2@cesar.school`)
  - Rafaela Vidal (`rbv2@cesar.school`)
  - Beatriz Pereira (`bcbsp@cesar.school`)
  - Manuela Cavalcanti (`mca2@cesar.school`)
  - Rafael Barros (`ralb@cesar.school`)
- **Repositório Remoto:** `git@github.com:msb7-jpg/atividade-paiva-av2.git`
- **Documentação Externa Compartilhada:** _(Insira aqui o link do Google Docs da equipe, se houver)_

---

## 2. Mapa da Documentação (`docs/`)

Toda a fundamentação teórica, exigências de entrega e especificações de produto residem na pasta `docs/`:

| Arquivo | Papel e Conteúdo |
| :--- | :--- |
| [`docs/instrucoes-avaliacao.md`](docs/instrucoes-avaliacao.md) | Enunciado formal da 2ª Avaliação, calendário de apresentações dos grupos e princípios científicos requeridos. |
| [`docs/exemplo-metodologia-grupo.md`](docs/exemplo-metodologia-grupo.md) | Transcrição e análise da metodologia aprovada na TV pelo grupo de referência (MobileNetV2, HSV, Grad-CAM). |
| [`docs/prd.md`](docs/prd.md) | Documento de Requisitos de Produto (PRD) com mapeamento da solução, objetivos e caso de uso. |
| [`docs/especificacao-metodologica.md`](docs/especificacao-metodologica.md) | Formalização matemática das hipóteses ($H_0$, $H_1$), equações do Teste $t$ pareado e protocolo estocástico pareado ($N=30$). |

Além dos arquivos Markdown, o material de aula do professor sobre Vision Transformers e Modelos Multimodais está disponível em [`slides-aula.pdf`](slides-aula.pdf).

---

## 3. Requisitos Científicos e Regras Inegociáveis

1. **Princípio do Isolamento de Variável (*Ceteris Paribus*):**
   - Comparação estrita de $A + B + C$ versus $A + B + C_{alt}$.
   - **Técnica A (Fixa / Pré-processamento):** *Data Augmentation* estocástico ($224 \times 224$ px, Flip Horizontal $p=0.5$, Rotação $\pm 10^\circ$, Color Jitter).
   - **Técnica B (Fixa / Arquitetura e Explicabilidade):** Arquitetura ResNet-50 com explicabilidade visual Grad-CAM (`layer4`) e auditoria diagnóstica por LMM multimodal local (Gemma 4 em `localhost:8080`).
   - **Técnica C (Variável Independente):**
     - *Cenário 1 (Proposto):* Transfer Learning com pesos pré-treinados ImageNet-1k, otimizador Adam ($\eta = 10^{-4}$, weight decay $L_2 = 10^{-4}$).
     - *Cenário 2 (Alternativa / Controle Contrafactual):* Treinamento a partir do zero (*scratch*) com inicialização He Normal (Kaiming), submetido a 1 época de *Linear Warmup* seguida de decaimento suave por cosseno (*Cosine Annealing*) com taxa $\eta \in [10^{-6}, 3 \times 10^{-4}]$.
2. **Amostragem Pareada com $N = 30$ Sementes:**
   - Amostragem com $30$ pares de execuções ($60$ treinamentos completos).
   - Conjunto fixo de sementes aleatórias: $\mathcal{S} = \{1001, 1002, \dots, 1030\}$.
   - O sorteio de lotes e transformações na iteração $i$ é idêntico para ambos os cenários sob a semente $s_i$.
   - A base é o **Chest X-Ray Dataset** (4.077 imagens de treino: 1.104 normais e 2.973 pneumonia; conjunto de teste com 582 imagens independentes: 171 normais e 411 pneumonia, imutável em todas as 60 rodadas).
3. **Validação por Inferência Estatística Formal:**
   - Aplicação de Teste $t$ de Student pareado bilateral com $\alpha = 0{,}05$.
   - Rejeição da Hipótese Nula ($H_0: \mu_1 - \mu_2 = 0$) em favor da Hipótese Alternativa ($H_1: \mu_1 - \mu_2 \neq 0$) caso $p < 0{,}05$ (resultado empírico: $p = 5{,}63 \times 10^{-11}$).
4. **Artigo Científico IEEE (Coluna Dupla):**
   - Redigido em padrão IEEE de conferência (2 colunas, 4 páginas).
   - **Regra Acadêmica Obrigatória:** Todo e qualquer parágrafo contém pelo menos uma citação formal de artigo seminal (NeurIPS, ICML, ICLR, CVPR, ICCV, JMLR).
   - Estilo de referências estritamente numérico (`[1]`, `[2]`) com tipografia `\scriptsize` e seção `\section*{References}`.

---

## 4. Arquitetura da Codebase (`src/`)

O código segue os princípios de **Clean Architecture** (adaptados de `.agents/skills/backend-coding-standards/`):

```
paiva-av2/
├── docs/                # Documentações e especificações científicas
├── latex/               # Template IEEEtran oficial, CSL e referências BibTeX
│   ├── ieee.csl         # Estilo de citação numérico IEEE
│   ├── referencias.bib  # Artigos seminais com BibTeX formal
│   └── template-ieee.tex# Template LaTeX customizado para Pandoc
├── paper/               # Fonte do artigo e saída PDF
│   ├── artigo.md        # Artigo em Markdown
│   ├── artigo.pdf       # Artigo compilado final
│   └── pages/           # Imagens PNG de inspeção visual das páginas
├── results/             # Gráficos de alta resolução e JSON de simulações
│   ├── boxplot_comparativo.png
│   ├── explicabilidade_xai.png
│   └── simulacoes_pareadas.json
├── scripts/             # Scripts executáveis diretos (sem Jupyter)
│   ├── build_paper.sh   # Compilação do artigo e geração de PNGs
│   └── run_simulation.sh# Execução direta da simulação pareada
├── src/                 # Código-fonte modular
│   ├── domain/          # Modelos Pydantic imutáveis
│   │   └── models.py
│   ├── services/        # Regras de negócio e inferência estatística pura
│   │   ├── experiment_runner.py
│   │   └── statistical_service.py
│   ├── plots/           # Geradores de figuras científicas
│   │   ├── generator.py
│   │   └── xai_visualizer.py
│   └── cli.py           # Interface de Linha de Comando (CLI com Typer & Rich)
├── pyproject.toml       # Dependências e metadados gerenciados com uv
└── PROGRESS.md          # Rastreabilidade de status e fases do projeto
```

---

## 5. Pré-Requisitos do Sistema e Instalação

Para que qualquer membro da equipe ou ambiente CI/CD execute os scripts, os seguintes pacotes devem estar disponíveis no sistema:

### A. Ferramentas de Compilação do Artigo (Pandoc e LaTeX)
- **Pandoc** ($\ge 3.0$): Conversor universal de documentos.
  - Download / Guia oficial: [https://pandoc.org/installing.html](https://pandoc.org/installing.html)
  - Ubuntu/Debian: `sudo apt install pandoc` ou via Homebrew: `brew install pandoc`
- **TeX Live / pdfLaTeX**: Motor LaTeX contendo a classe `IEEEtran.cls`.
  - Ubuntu/Debian:
    ```bash
    sudo apt install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended poppler-utils
    ```
  - *Nota:* O pacote `poppler-utils` fornece o utilitário `pdftoppm`, utilizado para renderizar automaticamente as páginas do artigo em PNG para inspeção visual.

### B. Ambiente Python e Dependências
- **Python** $\ge 3.12$
- **uv** (Gerenciador ultrarrápido de pacotes Python):
  - Guia oficial: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
  - Instalação rápida:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### C. Como Rodar o Projeto

1. **Clonar o Repositório:**
   ```bash
   git clone git@github.com:msb7-jpg/atividade-paiva-av2.git
   cd atividade-paiva-av2
   ```

2. **Executar a Simulação Pareada (Gera Estatísticas e Boxplot):**
   ```bash
   ./scripts/run_simulation.sh
   ```
   *O script cria o ambiente virtual isolado via `uv`, instala as dependências do `pyproject.toml`, executa as 30 simulações pareadas, exibe a tabela estatística no terminal e salva o gráfico em `results/boxplot_comparativo.png`.*

3. **Compilar o Artigo Científico IEEE em PDF:**
   ```bash
   ./scripts/build_paper.sh
   ```
   *O script invoca o Pandoc com o template IEEE, compila via pdfLaTeX com o arquivo `.bib` e o estilo `ieee.csl`, gerando `paper/artigo.pdf` e as imagens de inspeção em `paper/pages/`.*

---

## 6. Diretrizes Operacionais para Agentes de IA

Quando um agente for instruído a fazer alterações neste projeto, ele **deve** seguir as seguintes regras:

1. **Rastreabilidade Contínua:** Atualizar o arquivo [`PROGRESS.md`](PROGRESS.md) a cada conclusão de etapa.
2. **Commits Semânticos (Conventional Commits):**
   - `feat:` para novas rotinas experimentais, cálculos ou recursos de código.
   - `docs:` para atualizações de documentação, instruções ou texto do artigo.
   - `style:` para refinamentos de formatação LaTeX, tabelas ou layouts visuais.
   - `build:` para alterações de dependências no `pyproject.toml`, scripts de build ou templates.
3. **Validação de Build Obrigatória:** Sempre que alterar `paper/artigo.md`, `latex/template-ieee.tex` ou `latex/referencias.bib`, o agente deve executar obrigatoriamente `./scripts/build_paper.sh` e verificar se o PDF compilou sem erros.
4. **Isolamento de Scripts Pessoais:** Utilitários de infraestrutura pessoal ou de rede local não devem ser comitados no repositório compartilhado com outros membros.
