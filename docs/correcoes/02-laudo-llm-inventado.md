# Correção 2 — Laudo "inventado" quando o LLM não responde

> **Resumo:** quando o servidor do LLM (Gemma 4 em `localhost:8080`) não respondia, o serviço de laudos devolvia um **texto pronto, sempre igual**, que parecia uma análise da imagem. Os 9 laudos salvos em `results/` são esse texto: o LLM nunca rodou. A correção faz o serviço **falhar explicitamente** e grava um aviso claro de "laudo não gerado" no lugar do laudo falso.

Arquivos alterados: [`src/services/lmm_insights_service.py`](../../src/services/lmm_insights_service.py), [`src/plots/xai_visualizer.py`](../../src/plots/xai_visualizer.py), [`src/plots/gallery_generator.py`](../../src/plots/gallery_generator.py), [`src/cli.py`](../../src/cli.py).

---

## 1. Contexto: para que serve o LLM no projeto

Na **Técnica B** do estudo, a explicabilidade tem duas etapas:

1. **Grad-CAM** ([`gradcam_service.py`](../../src/services/gradcam_service.py)) gera um mapa de calor mostrando **onde** a ResNet-50 "olhou" para decidir. Ele usa os gradientes da última camada convolucional (`layer4`): as regiões cujas ativações mais aumentam a nota da classe prevista ficam "quentes" (vermelho).
2. **LMM** (*Large Multimodal Model*, um LLM que também enxerga imagens) recebe o raio-X original e o mapa de calor e escreve um "laudo" avaliando se a atenção do modelo faz sentido clínico: o foco está no pulmão ou em artefatos, como bordas, letras ou ossos?

O LMM roda **localmente** com o `llama-server` (do projeto llama.cpp), que expõe uma API **compatível com a da OpenAI**. Por isso o código conversa com ele usando o mesmo formato JSON da API do ChatGPT, só que apontando para `http://127.0.0.1:8080/v1`.

Fluxo de chamadas:

```
python -m src.cli xai      → xai_visualizer.generate_xai_and_insights_plot()  ─┐
python -m src.cli gallery  → gallery_generator.generate_clinical_gallery()    ─┼→ LMMInsightsService.analyze_radiograph_and_heatmap()
python -m src.cli run      → (etapa 5, se não usar --skip-lmm) xai_visualizer ─┘          │
                                                                                          ▼
                                                                         POST localhost:8080/v1/chat/completions
```

---

## 2. Como funciona o `lmm_insights_service.py` (arquivo completo)

### 2.1 Construtor

```python
class LMMInsightsService:
    def __init__(self, base_url="http://127.0.0.1:8080/v1",
                 model="unsloth/gemma-4-E4B-it-GGUF", timeout=60.0):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.model = model
        self.timeout = timeout
```

- `127.0.0.1` é o próprio computador (localhost). Nada sai para a internet.
- `rstrip("/")` remove uma barra final para não gerar `.../v1//chat/completions`.
- `/chat/completions` é o endpoint padrão do formato OpenAI para conversas.
- `timeout=60.0`: o LLM pode demorar para escrever, então damos até 60 s para a resposta.

### 2.2 Imagens em base64

```python
@staticmethod
def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
```

A requisição é um JSON, e JSON só carrega **texto**. Uma imagem é binária, então ela é:

1. salva como JPEG **na memória** (`BytesIO` funciona como um arquivo que vive na RAM, sem tocar o disco);
2. convertida para **base64**, uma codificação que representa bytes usando só letras, números, `+` e `/`;
3. embutida como uma *data URL*: `data:image/jpeg;base64,<texto>`.

`@staticmethod` indica que o método não usa `self`; ele só está dentro da classe por organização.

### 2.3 O prompt e o payload

```python
prompt = (
    f"Você é um especialista em radiologia computacional. "
    f"O modelo de rede neural classificou este exame como: {predicted_class} "
    f"com confiança de {confidence * 100:.1f}%.\n\n"
    ...
    f"1. Foco Anatômico: A atenção incide sobre o pulmão ou bordas externas?\n"
    f"2. Coerência com {predicted_class}: O padrão coincide com áreas suspeitas?\n"
    f"3. Conclusão: O diagnóstico do classificador é plausível sob a ótica dos achados visuais?"
)

payload = {
    "model": self.model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{heat_b64}"}},
        ],
    }],
    "temperature": 0.2,
    "max_tokens": 1000,
    "chat_template_kwargs": {"thinking": False},
}
```

- `messages` é a conversa. Aqui há uma única mensagem do usuário, com **conteúdo multimodal**: um bloco de texto e duas imagens, na ordem citada no prompt (primeiro a original, depois o mapa de calor).
- `temperature: 0.2` deixa a resposta pouco aleatória, porque queremos um laudo consistente, não criativo.
- `max_tokens: 1000` é o limite de tamanho da resposta.
- `chat_template_kwargs: {"thinking": False}` é uma opção do llama-server para o Gemma 4 responder direto, sem gerar o "raciocínio" interno antes.

### 2.4 A chamada HTTP (versão corrigida)

```python
try:
    client_timeout = httpx.Timeout(self.timeout, connect=3.0)
    with httpx.Client(timeout=client_timeout) as client:
        response = client.post(self.chat_url, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
    raise LMMUnavailableError(f"LMM em {self.chat_url} não respondeu: {e!r}") from e

if not content or not content.strip():
    raise LMMUnavailableError(f"LMM em {self.chat_url} retornou resposta vazia.")
return content.strip()
```

- `httpx` é a biblioteca de HTTP. `httpx.Timeout(60, connect=3.0)` dá **3 s para conectar** (se o servidor está desligado, desiste rápido) e 60 s para o resto.
- `with httpx.Client(...) as client:` abre a conexão e garante que ela é fechada no fim, mesmo se der erro.
- `raise_for_status()` transforma respostas HTTP de erro (como 404 ou 500) em exceção. Sem isso, um erro do servidor passaria como "resposta".
- `data["choices"][0]["message"]["content"]` navega no JSON de resposta do formato OpenAI até o texto gerado.

---

## 3. O problema

### 3.1 O código antigo

```python
try:
    ...  # mesma chamada HTTP
    return content.strip()
except Exception as e:
    logger.warning(f"Servidor LMM local offline ou resposta não obtida ({e}). Gerando parecer estruturado de fallback.")
    return (
        f"1. Análise Anatômica: O mapa de calor Grad-CAM converge para as bases e terço inferior dos pulmões.\n"
        f"2. Coerência Clínica: O padrão de ativação coincide com áreas de aumento de densidade compatíveis com {predicted_class}.\n"
        f"3. Síntese do Laudo: Decisão com foco em estruturas intra-pulmonares, sem dependência de artefatos de fundo."
    )
```

Em qualquer falha, o serviço **devolvia esse texto como se fosse o laudo**.

### 3.2 Por que isso está errado

1. **Parece análise, mas não olha a imagem.** O texto afirma que a atenção "converge para as bases dos pulmões" e "não depende de artefatos", sem ter visto imagem nenhuma. A única parte variável é o nome da classe.
2. **O aviso some.** O `logger.warning` vai para o log, que normalmente ninguém lê. O arquivo `.txt` salvo e o painel do terminal mostram o texto como laudo legítimo.
3. **Ele confirma até os erros do classificador.** O objetivo do laudo é **auditar** o classificador. No caso 05 da galeria, uma criança **sadia** que o modelo classificou errado como pneumonia, o "laudo" afirma que o padrão é "compatível com PNEUMONIA". Um auditor que sempre concorda não audita nada.
4. **`except Exception` esconde bugs.** Ele captura **qualquer** erro, inclusive erros de programação (uma variável com nome errado, por exemplo). Todos viravam um laudo bonito, e o bug ficava invisível.

### 3.3 Evidências

- Os **9 laudos** salvos no repositório (`results/laudo_lmm_insights.txt` e os 8 de `results/gallery/*_laudo.txt`) contêm exatamente esse texto pronto. O LLM nunca respondeu em nenhuma execução que gerou esses arquivos.
- O artigo ([`paper/artigo.md`](../../paper/artigo.md), seção de auditoria qualitativa) cita como "transcrição formal" do LLM um texto que **não é** o de reserva e **não aparece** em nenhum arquivo do projeto. Não dá para saber de onde ele veio.

---

## 4. A correção

A ideia central: **o serviço não decide o que fazer quando falha; ele avisa que falhou.** Quem chama decide como apresentar a falha, e nunca inventando conteúdo.

### 4.1 Uma exceção específica e uma marca padronizada

```python
LAUDO_NAO_GERADO = "[LAUDO NÃO GERADO — LMM indisponível]"


class LMMUnavailableError(RuntimeError):
    """O servidor LMM local não respondeu ou devolveu uma resposta inválida."""
```

- `LAUDO_NAO_GERADO` é uma **constante**: o mesmo texto é usado para gravar o aviso e para reconhecê-lo depois (a CLI usa `startswith`). Ter uma única definição evita que as duas pontas fiquem diferentes.
- `LMMUnavailableError` é uma **exceção própria**. Herdar de `RuntimeError` a coloca na família de erros "de execução". Ter um tipo próprio permite capturar **só** esse problema (`except LMMUnavailableError`) sem engolir outros erros.

### 4.2 Capturar só os erros esperados

```python
except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
    raise LMMUnavailableError(f"LMM em {self.chat_url} não respondeu: {e!r}") from e
```

Cada tipo cobre uma falha real de comunicação:

| Exceção | Quando acontece |
|---|---|
| `httpx.HTTPError` | Servidor desligado (conexão recusada), tempo esgotado, resposta HTTP de erro (via `raise_for_status`) |
| `ValueError` | A resposta não é JSON válido (`response.json()` lança `JSONDecodeError`, que é subclasse de `ValueError`) |
| `KeyError`, `IndexError`, `TypeError` | O JSON veio com estrutura inesperada (falta `"choices"`, lista vazia, valor `null` onde se esperava um dicionário) |

Um erro de programação, como um `NameError`, **não está na lista**: ele estoura normalmente e aparece para quem está desenvolvendo.

- `{e!r}` usa a representação "técnica" do erro, por exemplo `ConnectError('[WinError 10061] ...')`, que é melhor para diagnosticar.
- `raise ... from e` encadeia as exceções: o traceback mostra o erro original como causa. Nada da informação se perde.

Resposta vazia também é tratada como falha. Um LLM que responde `""` não gerou um laudo.

### 4.3 Quem chama decide: `laudo_or_unavailable_notice`

```python
def laudo_or_unavailable_notice(lmm_service, original_image, heatmap_image,
                                predicted_class, confidence) -> str:
    """Obtém o laudo do LMM ou um aviso explícito de que ele não foi gerado (nunca um texto inventado)."""
    try:
        return lmm_service.analyze_radiograph_and_heatmap(
            original_image, heatmap_image, predicted_class=predicted_class, confidence=confidence
        )
    except LMMUnavailableError as e:
        logger.warning(str(e))
        return f"{LAUDO_NAO_GERADO} {e}"
```

Por que existir essa função em vez de cada chamador fazer o próprio `try/except`?

- Os **dois** lugares que pedem laudo (painel XAI e galeria) querem o mesmo comportamento: continuar gerando as imagens do Grad-CAM, que não dependem do LLM, e marcar o laudo como ausente. A função evita duplicar essa lógica.
- O serviço continua "estrito" (falha com exceção). Se no futuro alguém quiser que a falha **interrompa** o programa, basta chamar `analyze_radiograph_and_heatmap` diretamente.

### 4.4 Os chamadores

Em [`xai_visualizer.py`](../../src/plots/xai_visualizer.py) e [`gallery_generator.py`](../../src/plots/gallery_generator.py), a mudança é só trocar a chamada:

```python
# antes
laudo = lmm_service.analyze_radiograph_and_heatmap(raw_img, overlay_t, predicted_class=..., confidence=...)
# depois
laudo = laudo_or_unavailable_notice(lmm_service, raw_img, overlay_t, predicted_class=..., confidence=...)
```

O resto continua igual: o texto (laudo real ou aviso) é salvo no `.txt`, e na galeria o cabeçalho do arquivo ainda traz o diagnóstico real e as predições.

### 4.5 A CLI destaca a ausência

No comando `xai` ([`cli.py`](../../src/cli.py)):

```python
if res["laudo"].startswith(LAUDO_NAO_GERADO):
    console.print(Panel(res["laudo"], title="LMM indisponível", border_style="yellow", expand=False))
else:
    console.print(Panel(res["laudo"], title="Laudo Clínico Estruturado (LMM)", expand=False))
```

Um laudo ausente aparece em **amarelo**, com o título "LMM indisponível", e não com o título de laudo clínico.

---

## 5. Teste

Comando `xai` com os modelos treinados no teste rápido e **sem** servidor LLM ligado:

```
Inferência do Modelo Transfer Learning:
  • Classe Predita: PNEUMONIA
  • Confiança: 95.73%

Inferência do Modelo Scratch (Warmup):
  • Classe Predita: NORMAL
  • Confiança: 96.66%

┌──────────────── LMM indisponível ────────────────┐
│ [LAUDO NÃO GERADO — LMM indisponível] LMM em http://127.0.0.1:8080/v1/chat/completions
│ não respondeu: ConnectError('[WinError 10061] Nenhuma conexão pôde ser feita porque a
│ máquina de destino as recusou ativamente')
└──────────────────────────────────────────────────┘
```

O arquivo `laudo.txt` gravado contém o mesmo aviso. O painel de Grad-CAM (PNG) foi gerado normalmente.

Esse teste também mostrou, pela primeira vez na RX 9060 XT, o Grad-CAM real:

- **Transfer:** acertou PNEUMONIA, com a atenção concentrada no centro-esquerda do tórax, perto da região do coração e do hilo. Não é exatamente o "parênquima pulmonar" que o artigo descreve.
- **Scratch:** errou (disse NORMAL, um falso negativo), com a atenção espalhada pela parte superior do tórax.

> Esse teste é de antes das correções 3 e 4; o modelo Scratch muda depois delas.

---

## 6. O que ainda falta

1. **Laudos reais exigem o LLM rodando no Windows.** O [`llama_lifecycle.py`](../../src/services/llama_lifecycle.py) só funciona em Linux (`pkill`, caminho `~/workspace/llama.cpp/build/bin/llama-server`). No Windows, dá para usar a versão do llama.cpp para Windows com backend **Vulkan**, que funciona em GPU AMD, mais o modelo `gemma-4-E4B-it` em GGUF (~5 GB).
2. **Os 9 laudos falsos de `results/` continuam no repositório.** O certo é gerá-los de novo com o LLM funcionando ou removê-los. Eles não foram apagados por serem arquivos do grupo.
3. **A citação de laudo no artigo** precisa ser trocada por um laudo realmente gerado ou removida.

## 7. Lição geral

> **Um fallback nunca deve fabricar o resultado que falhou.** Fallback aceitável é aquele que deixa a ausência **visível**: uma mensagem de erro, um valor marcado como "não disponível", um painel de outra cor. Capture só as exceções que você sabe tratar (`except Exception` amplo esconde bugs) e preserve a causa original (`raise ... from e`).
