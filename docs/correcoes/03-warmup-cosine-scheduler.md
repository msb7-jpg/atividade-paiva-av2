# Correção 3 — O *Linear Warmup + Cosine Annealing* não fazia nada

> **Resumo:** o artigo descreve o cenário Scratch (C_alt) como "1 época de *Linear Warmup* seguida de decaimento por cosseno, de 3×10⁻⁴ até 10⁻⁶". No código, o **warmup nunca acontecia** (com qualquer número de épocas), e com 2 épocas (o padrão da CLI) o **cosseno também não agia**: o learning rate ficava parado em 3×10⁻⁴ do primeiro ao último batch. As 30 runs do repositório foram geradas assim. A correção reescreve o scheduler para atualizar o learning rate **a cada batch**.

Arquivos alterados: [`src/services/resnet_builder.py`](../../src/services/resnet_builder.py), [`src/services/experiment_runner.py`](../../src/services/experiment_runner.py).

---

## 1. Conceitos necessários

### 1.1 Época, batch e step

- O conjunto de treino tem **4077** radiografias.
- O modelo não vê as 4077 de uma vez. Elas são divididas em **batches** de 64 imagens: ⌈4077 / 64⌉ = **64 batches** (63 cheios + 1 com 45 imagens).
- Processar um batch e atualizar os pesos uma vez é um **step** (passo).
- Passar pelas 4077 imagens uma vez (64 steps) é uma **época**.
- Com 2 épocas, o treino tem 2 × 64 = **128 steps**.

### 1.2 Learning rate (lr)

A cada step, o otimizador ajusta os pesos na direção que reduz o erro (o gradiente). O **learning rate** controla o tamanho desse ajuste:

- lr alto demais → passos grandes, o treino "pula" e pode divergir;
- lr baixo demais → passos minúsculos, o treino quase não sai do lugar.

O projeto usa o otimizador **Adam** com `weight_decay=1e-4`. No `optim.Adam` do PyTorch, isso soma uma penalidade L2 ao gradiente, puxando os pesos levemente para zero (regularização).

### 1.3 Por que variar o lr durante o treino (o *scheduler*)

Um **scheduler** muda o lr ao longo do treino. O nosso combina duas ideias:

**Warmup linear (aquecimento):** começar com lr quase zero e subir em linha reta até o valor máximo (3×10⁻⁴). É importante principalmente **no treino do zero**:
- com pesos aleatórios, os primeiros gradientes são grandes e desorganizados;
- o Adam estima a "escala" de cada gradiente a partir do histórico, e no começo esse histórico tem poucos steps, então a estimativa é ruim;
- passos grandes nesse momento podem jogar a rede numa região ruim, de onde ela demora a sair.

**Cosine annealing (decaimento por cosseno):** depois do warmup, o lr desce suavemente seguindo meia onda de cosseno, do máximo até um mínimo (10⁻⁶). Passos cada vez menores no final deixam o modelo se acomodar num bom ponto, em vez de ficar oscilando em volta dele.

As fórmulas (s = step atual, W = steps de warmup, T = total de steps):

```
warmup  (s < W):   lr(s) = lr_max · (s + 1) / W

cosseno (s ≥ W):   p = (s − W) / (T − W − 1)                # progresso de 0 a 1
                   lr(s) = lr_min + ½ · (lr_max − lr_min) · (1 + cos(π · p))
```

Quando p = 0, cos(0) = 1 e lr = lr_max. Quando p = 1, cos(π) = −1 e lr = lr_min. No meio (p = 0,5), cos(π/2) = 0 e lr ≈ metade do máximo.

```
lr
3e-4 |         ________
     |        /        ‾‾-_
     |       /             ‾-_
     |      /                 ‾-_
     |     /                     ‾-_
1e-6 |____/                          ‾‾‾__
     +----------------------------------------> steps
     0    W=64                          T=128
       warmup            cosseno
```

---

## 2. O arquivo `resnet_builder.py` completo

O arquivo tem duas partes: a fábrica do modelo e o scheduler.

### 2.1 `create_resnet50`: cria o modelo dos dois cenários

```python
def create_resnet50(pretrained: bool = True, seed: int = 1001) -> nn.Module:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model
```

- `torch.manual_seed(seed)` fixa o gerador aleatório **global** do PyTorch, para que toda inicialização aleatória seja reproduzível na mesma seed. `cuda.manual_seed_all` faz o mesmo na GPU (no build ROCm, "cuda" é a GPU AMD).
- **Cenário 1 (Transfer):** `ResNet50_Weights.DEFAULT` carrega os pesos treinados no **ImageNet-1k** (a versão `IMAGENET1K_V2` do torchvision, arquivo `resnet50-11ad3fa6.pth`, ~98 MB, baixado na primeira execução).
- **Cenário 2 (Scratch):** `weights=None` cria a rede com pesos aleatórios. O laço percorre todas as camadas (`model.modules()`) e aplica:
  - **Kaiming/He normal** nas convoluções: sorteia pesos de uma normal com variância calibrada para redes com ReLU, de modo que o sinal não exploda nem suma ao atravessar as 50 camadas. `fan_out` usa o número de saídas da camada para essa calibração;
  - BatchNorm com escala 1 e deslocamento 0, ou seja, começa "neutro".
- **A última camada é trocada nos dois cenários:** a ResNet original termina com `fc` de 2048 → 1000 (as 1000 classes do ImageNet). Ela vira `nn.Linear(2048, 2)` (NORMAL/PNEUMONIA), com pesos novos e aleatórios.

> **Duas observações** (não alteram o comportamento):
> 1. O `resnet50(weights=None)` do torchvision **já aplica** Kaiming normal `fan_out` e BatchNorm 1/0 por padrão, então o laço é redundante. Ele só deixa a escolha explícita.
> 2. O `nn.init.normal_(…, 0.01)` do laço atinge a `fc` **antiga** (1000 classes), que é descartada logo depois. A `fc` nova de 2 classes usa a inicialização padrão do `nn.Linear` nos **dois** cenários, o que é bom para o *ceteris paribus*.

### 2.2 `LinearWarmupCosineAnnealingLR`: o scheduler (versão corrigida)

```python
class LinearWarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
        self._apply_lr()

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step + 1) / self.warmup_steps
        cosine_steps = max(1, self.total_steps - self.warmup_steps - 1)
        progress = min(1.0, (self.current_step - self.warmup_steps) / cosine_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        self.current_step += 1
        self._apply_lr()

    def _apply_lr(self) -> None:
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
```

Parte a parte:

- **Estado:** a classe guarda só `current_step`, o número de steps já dados. Todo o resto é configuração.
- **`max(1, warmup_steps)`** evita divisão por zero se alguém passar `warmup_steps=0`.
- **`_apply_lr()` no construtor** é a peça-chave da correção. Assim que o scheduler é criado, ele já grava no otimizador o lr do **step 0** (início do warmup). Sem isso, o primeiro batch usaria o lr com que o `Adam` foi criado (3×10⁻⁴, o máximo).
- **`get_lr()`** é a fórmula da seção 1.3, isolada numa função pura (só lê o estado e devolve um número), o que facilita testar:
  - o `+ 1` no warmup faz o step 0 já ter um lr positivo (lr_max/W) e o step W−1 chegar exatamente a lr_max;
  - `cosine_steps = T − W − 1` faz o **último** step (s = T−1) ter p = 1, ou seja, exatamente lr_min;
  - `min(1.0, …)` protege caso alguém chame `step()` além do total: o lr fica em lr_min em vez de voltar a subir (o cosseno é periódico).
- **`_apply_lr()`**: o otimizador do PyTorch guarda o lr dentro de `param_groups`, uma lista de grupos de parâmetros, cada um com sua configuração. Aqui todos os grupos recebem o mesmo lr.
- **`step()`**: avança um step e aplica o novo lr. Deve ser chamado **depois** de `optimizer.step()`, a mesma convenção dos schedulers nativos do PyTorch.

> O PyTorch tem schedulers prontos que fazem isso, como `LambdaLR` ou `SequentialLR(LinearLR, CosineAnnealingLR)`. A classe própria foi mantida para não mudar a estrutura do projeto e porque, sendo curta, é fácil de explicar na apresentação.

---

## 3. O laço de treino (`experiment_runner.py`)

Em `ExperimentRunner.train_and_evaluate`:

```python
model = create_resnet50(pretrained=pretrained, seed=seed).to(self.device)
criterion = nn.CrossEntropyLoss()

if pretrained:
    # Transfer Learning: lr constante e menor
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = None
else:
    # Scratch: Warmup (1 época) + Cosine, com o lr atualizado a cada batch
    steps_per_epoch = len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=1 * steps_per_epoch,
        total_steps=self.epochs * steps_per_epoch,
        base_lr=3e-4,
        min_lr=1e-6,
    )

model.train()
for epoch in range(self.epochs):
    for x, y in train_loader:
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        optimizer.zero_grad()      # 1. zera os gradientes do step anterior
        out = model(x)             # 2. forward: previsões (logits) do batch
        loss = criterion(out, y)   # 3. erro: entropia cruzada entre previsão e rótulo
        loss.backward()            # 4. backward: calcula o gradiente de cada peso
        optimizer.step()           # 5. atualiza os pesos usando o lr atual
        if scheduler:
            scheduler.step()       # 6. prepara o lr do PRÓXIMO step
```

- `len(train_loader)` é o número de batches por época (64). Assim, "1 época de warmup" vira 64 steps e o total vira `épocas × 64`.
- `.to(self.device)` copia o batch para a GPU. `non_blocking=True` permite que a cópia aconteça em paralelo com outras operações.
- `CrossEntropyLoss` é a perda padrão de classificação: ela pune mais quando o modelo dá alta confiança à classe errada.
- `zero_grad()` é necessário porque o PyTorch **acumula** gradientes por padrão.
- O Transfer **não usa scheduler** (`scheduler = None`): lr fixo de 10⁻⁴. Essa diferença entre cenários é discutida na [seção 6](#6-consequências-e-pendências) (ceteris paribus).

---

## 4. O problema

### 4.1 O código antigo

O scheduler contava **épocas**:

```python
class LinearWarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        ...
        self.current_epoch = 0            # repare: o construtor NÃO mexia no lr

    def step(self) -> None:
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
```

E o laço chamava `step()` **fora** do laço de batches, uma vez por época:

```python
for epoch in range(self.epochs):
    for x, y in train_loader:
        ...
        optimizer.step()
    if scheduler:
        scheduler.step()      # ← só no FIM de cada época
```

### 4.2 Execução passo a passo (2 épocas, o padrão da CLI)

| Momento | O que acontece | lr usado |
|---|---|---|
| Criação | `Adam(lr=3e-4)`. O scheduler é criado mas **não altera** o lr | — |
| Época 0 (64 batches) | Treina | **3×10⁻⁴** (o valor do Adam: não houve warmup) |
| Fim da época 0 | `step()`: epoch=1 ≤ warmup=1 → lr = 3e-4 × 1/1 | — |
| Época 1 (64 batches) | Treina | **3×10⁻⁴** |
| Fim da época 1 | `step()`: epoch=2 → p = 1 → lr = 10⁻⁶ | — (o treino já acabou) |

Dois defeitos se somam:

1. **O warmup nunca acontecia.** O primeiro valor que o scheduler calculava (época 1 de 1 de warmup) já era o lr máximo, e a época 0 usava o valor do Adam, também o máximo. Isso vale **para qualquer número de épocas**.
2. **O cosseno só calculava o lr depois de usado.** O valor de cada época era aplicado ao fim da época anterior. Com 2 épocas, o único valor do cosseno (10⁻⁶) foi calculado depois do último batch.

### 4.3 Medição

Um script simulou o mesmo laço e registrou o lr em vigor em cada época:

```
epochs=2: lr usado em cada época = ['3.00e-04', '3.00e-04']
epochs=5: lr usado em cada época = ['3.00e-04', '3.00e-04', '2.56e-04', '1.50e-04', '4.48e-05']
```

Com 2 épocas, o lr ficou **constante**. Com 5, o cosseno aparece, mas em degraus de época e **ainda sem warmup**.

### 4.4 Por que isso importa

- **O artigo descreve um treino que não foi executado.** O método do cenário C_alt ("1 época de Linear Warmup + Cosine Annealing") não corresponde ao que gerou as 30 runs. Num trabalho cujo foco é o rigor experimental, isso é grave.
- **O Scratch foi treinado em condições piores que as descritas.** Começar do zero com lr cheio é exatamente o que o warmup existe para evitar.

---

## 5. A correção e a verificação

As mudanças (todas mostradas nas seções 2.2 e 3):

1. O scheduler passou a contar **steps (batches)** em vez de épocas.
2. O construtor **já aplica** o lr do step 0.
3. `scheduler.step()` foi para **dentro** do laço de batches, logo após `optimizer.step()`.
4. O runner converte "1 época de warmup" e "N épocas no total" em steps com `len(train_loader)`.

Verificação com 64 batches por época:

```
epochs=2 (128 batches)
  batch   0 (época 0): lr = 4.69e-06     ← início do warmup (3e-4 / 64)
  batch   1 (época 0): lr = 9.37e-06
  batch  31 (época 0): lr = 1.50e-04     ← metade do warmup
  batch  63 (época 0): lr = 3.00e-04     ← pico
  batch  64 (época 1): lr = 3.00e-04     ← início do cosseno
  batch  65 (época 1): lr = 3.00e-04
  batch  96 (época 1): lr = 1.47e-04     ← metade do cosseno
  batch 126 (época 1): lr = 1.19e-06
  batch 127 (época 1): lr = 1.00e-06     ← último batch = lr_min exato

epochs=5 (320 batches)
  batch   0 (época 0): lr = 4.69e-06
  batch  63 (época 0): lr = 3.00e-04
  batch 192 (época 3): lr = 1.50e-04
  batch 319 (época 4): lr = 1.00e-06
```

### 5.1 Treino real de validação (seed 1001, 2 épocas, RX 9060 XT)

Treino completo com as correções 3 **e** 4 (pareamento) aplicadas juntas:

| Seed 1001 | Transfer | Scratch (antes das correções 3/4) | Scratch (depois das correções 3/4) |
|---|---|---|---|
| F1-macro | 0,9464 | 0,8777 | 0,8138 |
| Acurácia | 0,9570 | 0,8918 | 0,8643 |
| Falsos negativos | 3 | 53 | 8 |
| Falsos positivos | 22 | 10 | 71 |

- O **Transfer não mudou em nada**, como esperado: ele não usa scheduler, e o loader dele já começava do gerador "zerado" (veja a Correção 4). A coincidência exata também mostra que o treino é reproduzível entre execuções.
- O **Scratch mudou bastante**: piorou no F1 e trocou o tipo de erro (de deixar passar pneumonia para gerar alarmes falsos). Como as duas correções foram aplicadas juntas, não dá para atribuir a mudança a uma só. E **uma única seed não permite conclusões**: só as 30 seeds dizem se o Scratch melhora ou piora em média.

---

## 6. Consequências e pendências

1. **As 30 runs precisam ser refeitas** para que os resultados correspondam ao método descrito no artigo.
2. **O número de épocas está inconsistente e não é informado no artigo:**
   - CLI (`python -m src.cli run`): **2** épocas (padrão de `--epochs`);
   - `ExperimentRunner` usado diretamente: **5** épocas (padrão do construtor);
   - `xai` sem checkpoint: treina **1** época.

   Duas épocas é pouco, especialmente para treinar do zero, e com 2 épocas o warmup ocupa **metade** do treino. É uma decisão do grupo, ligada à **Correção 5 (ceteris paribus)**: hoje C e C_alt diferem na inicialização **e** no otimizador (lr 10⁻⁴ fixo vs 3×10⁻⁴ com scheduler), o que mistura duas variáveis independentes.

## 7. Lição geral

> **Verifique o hiperparâmetro que o código realmente usa, não o que ele declara usar.** Um scheduler mal posicionado no laço não dá erro nenhum: o treino roda, converge e gera números plausíveis. Só medindo (registrar o lr de cada step, plotar a curva) é que o problema aparece. Na dúvida, prefira atualizar o scheduler por step e conferir o lr do primeiro e do último batch.
