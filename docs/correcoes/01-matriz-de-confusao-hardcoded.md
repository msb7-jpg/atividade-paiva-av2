# Correção 1 — Matriz de confusão escrita à mão no `status`

> **Resumo:** o comando `python -m src.cli status` exibia uma matriz de confusão (537/582 acertos no Transfer, 500/582 no Scratch) que estava **digitada como texto fixo no código**, não lida dos resultados. Esses números não batem com os dados salvos nem com um treino real. A correção faz o painel ser montado a partir dos arquivos JSON de cada run e avisar quando a run não tem a matriz.

Arquivos alterados: [`src/domain/models.py`](../../src/domain/models.py), [`src/cli.py`](../../src/cli.py).

---

## 1. Conceitos necessários

### 1.1 O que é a matriz de confusão

Nosso problema é de **classificação binária**: cada radiografia é `NORMAL` (classe 0) ou `PNEUMONIA` (classe 1). Tratamos **pneumonia como a classe "positiva"**, ou seja, a doença que queremos detectar.

Depois de treinar, o modelo classifica as 582 imagens do conjunto de teste e comparamos a previsão com o rótulo verdadeiro. Cada imagem cai em um de quatro "quadrantes":

|                         | Modelo diz NORMAL | Modelo diz PNEUMONIA |
|-------------------------|:-----------------:|:--------------------:|
| **Real: NORMAL** (171)    | **TN** – Verdadeiro Negativo | **FP** – Falso Positivo (alarme falso) |
| **Real: PNEUMONIA** (411) | **FN** – Falso Negativo (doença não detectada) | **TP** – Verdadeiro Positivo |

Toda métrica clínica sai desses 4 números:

| Métrica | Fórmula | Pergunta que responde |
|---|---|---|
| Acurácia | (TP + TN) / total | De todas as imagens, quantas o modelo acertou? |
| **Sensibilidade** (recall da pneumonia) | TP / (TP + FN) | Das crianças **com** pneumonia, quantas o modelo detectou? |
| **Especificidade** (recall do normal) | TN / (TN + FP) | Das crianças **sadias**, quantas o modelo confirmou como sadias? |
| F1-macro | média do F1 das duas classes | Equilíbrio entre precisão e recall, dando o mesmo peso às duas classes (importante porque o teste é desbalanceado: 411 vs 171) |

Na medicina, **FN costuma ser o erro mais grave**: uma criança doente vai para casa sem tratamento. Um FP gera um exame extra, que é chato mas não perigoso. Por isso a sensibilidade e a especificidade contam mais do que a acurácia sozinha.

### 1.2 Pydantic (usado em `models.py`)

O projeto usa a biblioteca **Pydantic** para definir as estruturas de dados. Uma classe que herda de `BaseModel` declara campos com tipos, e o Pydantic:

- **valida** os dados ao criar o objeto. Por exemplo, `Field(ge=0.0, le=1.0)` faz `accuracy=1.3` gerar um erro;
- **converte para dicionário/JSON** com `.model_dump()`, que é o que gravamos em disco;
- **reconstrói o objeto** a partir de um dicionário com `Classe(**dados)`, que é o que fazemos ao ler o JSON.

---

## 2. Como a matriz é calculada, salva e lida (o caminho completo)

### 2.1 Cálculo: `ExperimentRunner.train_and_evaluate` ([`experiment_runner.py`](../../src/services/experiment_runner.py))

Depois do treino, o modelo é avaliado no conjunto de teste:

```python
model.eval()                          # desliga comportamentos de treino (BatchNorm usa as estatísticas acumuladas)
all_preds: list[int] = []
all_targets: list[int] = []

with torch.no_grad():                 # não calcula gradientes: economiza memória e tempo
    for x, y in test_loader:          # percorre as 582 imagens em batches de 64
        x = x.to(self.device, non_blocking=True)
        out = model(x)                # "logits": 2 números por imagem (um por classe)
        preds = torch.argmax(out, dim=1).cpu().numpy()   # a classe com maior logit vira a previsão
        all_preds.extend(preds)
        all_targets.extend(y.numpy())
```

Em seguida, a matriz:

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
```

O `confusion_matrix` do scikit-learn devolve uma matriz 2×2 em que a **linha é a classe real** e a **coluna é a classe prevista**. `labels=[0, 1]` fixa a ordem (0 = NORMAL, 1 = PNEUMONIA). Então:

- `cm[0,0]`: real 0, previsto 0 → **TN**
- `cm[0,1]`: real 0, previsto 1 → **FP**
- `cm[1,0]`: real 1, previsto 0 → **FN**
- `cm[1,1]`: real 1, previsto 1 → **TP**

O `int(...)` converte de `numpy.int64` para `int` do Python, que o Pydantic e o JSON aceitam sem problemas.

As métricas derivadas protegem contra divisão por zero (`if (tp + fn) > 0 else 0.0`) e são arredondadas para 4 casas antes de ir para o objeto `EvaluationMetrics`.

### 2.2 Gravação: `run_single_seed`

Cada seed gera um `PairedSimulationResult` (cenário 1 + cenário 2), gravado imediatamente em `results/runs/seed_XXXX.json`:

```python
with open(run_file, "w", encoding="utf-8") as f:
    json.dump(paired.model_dump(), f, indent=2)
```

Um JSON gerado pelo código atual fica assim (teste real na seed 1001, RX 9060 XT):

```json
{
  "seed": 1001,
  "scenario_1": {
    "f1_macro": 0.9464, "accuracy": 0.957,
    "total_samples": 582, "correct_count": 557, "error_count": 25,
    "tp": 408, "tn": 149, "fp": 22, "fn": 3,
    "sensitivity": 0.9927, "specificity": 0.8713
  },
  "scenario_2": { "...": "..." }
}
```

### 2.3 Leitura: `ExperimentRunner.get_all_saved_runs`

O `status` chama esse método, que abre cada `seed_*.json` e reconstrói o objeto com `PairedSimulationResult(**json.load(f))`. É nesse momento que o Pydantic valida e preenche os campos.

---

## 3. O problema

### 3.1 O código antigo

No [`cli.py`](../../src/cli.py), o comando `status` fazia:

```python
if 1001 in completed_seeds:
    cm_panel = (
        "...Cenário 1 (Transfer Learning - ResNet-50 ImageNet):\n"
        "  • Acertos Totais: [bold]537 / 582 (92.27%)[/bold]  |  Erros: [bold]45 / 582 (7.73%)[/bold]\n"
        "  • Sensibilidade (Pneumonia): [bold]99.27%[/bold] (408 de 411 detectadas, apenas 3 Falsos Negativos)\n"
        "  • Especificidade (Sadio):   [bold]75.44%[/bold] (129 de 171 confirmadas sadias)\n\n"
        "...Cenário 2 (Scratch - Treinamento do Zero + Cosine):\n"
        "  • Acertos Totais: [bold]500 / 582 (85.91%)[/bold]  |  Erros: [bold]82 / 582 (14.09%)[/bold] ...\n"
        ...
    )
```

A única coisa que vinha dos dados era a **condição** `if 1001 in completed_seeds`, que só verifica se o arquivo existe. **Todo o conteúdo era texto fixo.** Treinar outro modelo, mudar o dataset ou apagar o checkpoint não mudava em nada o que aparecia na tela.

### 3.2 Evidências de que os números estão errados

1. **Não fecham com o JSON da própria seed.** O `results/runs/seed_1001.json` do repositório diz `"accuracy": 0.9605` para o Transfer. Como 0,9605 × 582 ≈ **559** acertos, a matriz não pode ter 537 (que dá 92,27%).
2. **Não batem com um treino real.** Treinando a seed 1001 na RX 9060 XT, saíram **557/582** (Transfer) e **519/582** (Scratch), contra os 537 e 500 do texto fixo.
3. **Nunca foram calculados a partir das 30 runs.** No histórico do git, um mesmo commit (`4ec1fec`) fez três coisas:
   - adicionou ao `experiment_runner.py` o código que calcula TP/TN/FP/FN;
   - adicionou os 30 JSONs, que **só têm `f1_macro` e `accuracy`**, sem os campos da matriz;
   - escreveu "537/582" no `cli.py`, no artigo, no README, nos slides e nos docs.

   Se os JSONs tivessem saído do código novo, teriam os campos `tp`, `tn` etc. Ou seja, os JSONs vieram de uma versão anterior do código e a matriz foi escrita à parte. A origem dos números não pode ser verificada.

### 3.3 Um segundo bug, escondido no modelo de dados

Antes da correção, os campos da matriz tinham valores padrão:

```python
total_samples: int = Field(default=582, ...)
correct_count: int = Field(default=0, ...)
tp: int = Field(default=0, ...)
# ... tn, fp, fn = 0; sensitivity, specificity = 0.0
```

Isso parece inofensivo, mas quando o Pydantic lê um JSON **antigo** (sem esses campos), ele os preenche com os padrões. O resultado é um objeto que diz *"582 imagens, 0 acertos, sensibilidade 0%"*. Não há como distinguir **"não foi medido"** de **"foi medido e deu zero"**. Qualquer código que tentasse ler a matriz de verdade mostraria zeros como se fossem resultado real.

### 3.4 Por que isso importa para o trabalho

A Tabela II do artigo, o README, os slides e a parte do Ygor na apresentação constroem uma **narrativa clínica** em cima desses números ("o Scratch colapsa na especificidade"). Mas o perfil de erro do Scratch numa seed isolada é muito instável. Veja a mesma seed 1001 treinada de verdade, antes e depois das correções 3 e 4 (scheduler e pareamento):

| Seed 1001, Scratch | Texto fixo (docs/artigo) | Treino real, antes das correções 3/4 | Treino real, depois das correções 3/4 |
|---|---|---|---|
| Acertos | 500/582 | 519/582 | 503/582 |
| Falsos negativos | 7 | **53** | 8 |
| Falsos positivos | 75 | 10 | **71** |
| Perfil | "colapsa na especificidade" | deixa passar pneumonia | colapsa na especificidade |

Detalhes do treino (ordem dos batches, learning rate) **invertem completamente** o tipo de erro do Scratch. Uma única seed não sustenta nenhuma narrativa clínica, nem a do artigo nem a contrária. Por isso a tabela tem que ser a **média das 30 seeds**, calculada a partir dos dados.

---

## 4. A correção

### 4.1 `models.py`: "não medido" passa a ser `None`

```python
class EvaluationMetrics(BaseModel):
    f1_macro: float = Field(..., ge=0.0, le=1.0, description="F1-Score macro")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Acurácia geral")
    # Campos da matriz de confusão: None = não medido (runs antigas salvas só com F1/acurácia)
    total_samples: int | None = Field(default=None, ...)
    correct_count: int | None = Field(default=None, ...)
    error_count: int | None = Field(default=None, ...)
    tp: int | None = Field(default=None, ...)
    tn: int | None = Field(default=None, ...)
    fp: int | None = Field(default=None, ...)
    fn: int | None = Field(default=None, ...)
    sensitivity: float | None = Field(default=None, ge=0.0, le=1.0, ...)
    specificity: float | None = Field(default=None, ge=0.0, le=1.0, ...)

    @property
    def has_confusion_matrix(self) -> bool:
        return None not in (self.tp, self.tn, self.fp, self.fn)
```

Linha a linha:

- `f1_macro` e `accuracy` continuam **obrigatórios** (`Field(...)`, onde as reticências significam "sem padrão"). Toda run tem essas duas métricas.
- `int | None` diz que o campo pode ser um inteiro **ou** ausente. O padrão agora é `None`, que significa "esta run não mediu isso".
- `ge=0.0, le=1.0` continua valendo para sensibilidade/especificidade. No Pydantic v2, a restrição só é verificada quando o valor não é `None`.
- `has_confusion_matrix` é uma `@property`: lê-se como atributo (`m.has_confusion_matrix`, sem parênteses), mas é calculada na hora. Ela responde "esta run tem os 4 números?".

**Compatibilidade:** os JSONs antigos continuam carregando sem erro (os campos ficam `None`), e os novos carregam com os valores reais. Ninguém do grupo precisa apagar nada.

### 4.2 `cli.py`: o painel é montado a partir dos dados

Uma função nova formata a matriz de um cenário:

```python
def _format_confusion_matrix(header: str, m: EvaluationMetrics) -> str:
    """Formata a matriz de confusão de um cenário a partir das métricas salvas em disco."""
    tp, tn, fp, fn = m.tp, m.tn, m.fp, m.fn
    assert tp is not None and tn is not None and fp is not None and fn is not None
    positives = tp + fn          # total de pneumonias reais (411)
    negatives = tn + fp          # total de sadios reais (171)
    total = positives + negatives
    correct = tp + tn
    errors = fp + fn
    return (
        f"{header}\n"
        f"  • Acertos Totais: [bold]{correct} / {total} ({correct / total:.2%})[/bold]  |  "
        f"Erros: [bold]{errors} / {total} ({errors / total:.2%})[/bold]\n"
        f"  • Sensibilidade (Pneumonia): [bold]{tp / positives:.2%}[/bold] "
        f"({tp} de {positives} detectadas, {fn} Falsos Negativos)\n"
        f"  • Especificidade (Sadio):   [bold]{tn / negatives:.2%}[/bold] "
        f"({tn} de {negatives} confirmadas sadias, {fp} Falsos Positivos)"
    )
```

Detalhes:

- O `_` no início do nome é a convenção do Python para função "interna" do módulo, que não faz parte da interface da CLI.
- **Tudo é recalculado a partir dos 4 números.** Total, acertos, sensibilidade e especificidade saem de TP/TN/FP/FN, então o painel nunca pode ficar internamente inconsistente (por exemplo, acurácia que não bate com os acertos, como acontecia antes).
- `{x:.2%}` é uma formatação do Python que multiplica por 100 e mostra com 2 casas e o símbolo `%`. Por exemplo, `0.9570` vira `95.70%`.
- Os trechos `[bold]...[/bold]` são marcações da biblioteca **Rich** (negrito no terminal).
- **Por que o `assert`?** Como os campos agora são `int | None`, o verificador de tipos do editor (Pylance) reclamaria de `tp + fn`, porque `None + int` não existe. O `assert ... is not None` documenta a pré-condição (quem chama já verificou `has_confusion_matrix`) e faz o verificador entender que dali em diante são `int`. Se alguém chamar a função errado, ela falha na hora, com uma mensagem clara, em vez de imprimir lixo.

E o trecho do `status` que usa a função:

```python
ref_seed = 1001
if ref_seed in completed_seeds:
    ref_run = completed_seeds[ref_seed]
    if ref_run.scenario_1.has_confusion_matrix and ref_run.scenario_2.has_confusion_matrix:
        cm_panel = (
            _format_confusion_matrix("...Cenário 1 (Transfer Learning ...):", ref_run.scenario_1)
            + "\n\n"
            + _format_confusion_matrix("...Cenário 2 (Scratch ...):", ref_run.scenario_2)
        )
    else:
        cm_panel = (
            "[yellow]Esta run foi salva sem a matriz de confusão (apenas F1 e acurácia).[/yellow]\n"
            f"Reexecute com: python -m src.cli run --start-seed {ref_seed} --end-seed {ref_seed} --no-resume"
        )
    console.print(Panel(cm_panel, title=f"Auditoria Clínica Amostral (Semente de Referência {ref_seed})", expand=False))
```

São três caminhos possíveis:

1. A seed 1001 não foi executada: não mostra painel nenhum (igual a antes).
2. Foi executada e tem a matriz: mostra os números **lidos do JSON**.
3. Foi executada, mas é uma run antiga sem matriz: mostra um aviso amarelo com o comando exato para gerar a matriz. Nada é inventado.

---

## 5. Teste

**Com os JSONs antigos do repositório** (`python -m src.cli status`):

```
┌───────────── Auditoria Clínica Amostral (Semente de Referência 1001) ──────────────┐
│ Esta run foi salva sem a matriz de confusão (apenas F1 e acurácia).                │
│ Reexecute com: python -m src.cli run --start-seed 1001 --end-seed 1001 --no-resume │
└────────────────────────────────────────────────────────────────────────────────────┘
```

**Com o JSON do treino real** (`python -m src.cli status -d <pasta do teste>`):

```
┌───────────────── Auditoria Clínica Amostral (Semente de Referência 1001) ─────────────────┐
│ Cenário 1 (Transfer Learning - ResNet-50 ImageNet):                                       │
│   • Acertos Totais: 557 / 582 (95.70%)  |  Erros: 25 / 582 (4.30%)                        │
│   • Sensibilidade (Pneumonia): 99.27% (408 de 411 detectadas, 3 Falsos Negativos)         │
│   • Especificidade (Sadio):   87.13% (149 de 171 confirmadas sadias, 22 Falsos Positivos) │
│                                                                                           │
│ Cenário 2 (Scratch - Treinamento do Zero + Cosine):                                       │
│   • Acertos Totais: 519 / 582 (89.18%)  |  Erros: 63 / 582 (10.82%)                       │
│   • Sensibilidade (Pneumonia): 87.10% (358 de 411 detectadas, 53 Falsos Negativos)        │
│   • Especificidade (Sadio):   94.15% (161 de 171 confirmadas sadias, 10 Falsos Positivos) │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

> Esse resultado é de **antes** das correções 3 e 4 (scheduler e pareamento), que mudam o treino. Ele serve para validar o painel, não como resultado final.

---

## 6. O que ainda falta

1. **Rodar as 30 seeds de novo.** Os JSONs atuais não têm a matriz, e isso não dá para recuperar. Como as correções 3 e 4 também mudam o treino, o ideal é rodar uma vez só depois de todas as correções (~3h20 na RX 9060 XT).
2. **Gerar a Tabela II do artigo a partir dos dados.** O ideal é usar a **média ± desvio das 30 seeds**, e não uma seed escolhida: a seed 1001 pode ser um caso extremo.
3. **Atualizar artigo, README, slides, `docs/prd.md` e `docs/especificacao-metodologica.md`**, que ainda citam 537/500.

## 7. Lição geral

> **Número que aparece em relatório tem que sair dos dados, nunca ser digitado.** Se o dado não existe, o certo é mostrar que ele está ausente ("não medido"), e não preencher com um valor plausível. Isso vale tanto para texto fixo no código quanto para valores padrão "inocentes", como `default=0`.
