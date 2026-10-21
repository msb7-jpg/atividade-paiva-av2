# Correção 4 — O pareamento entre os cenários estava quebrado

> **Resumo:** o protocolo diz que, em cada seed, o Transfer (C) e o Scratch (C_alt) treinam com **exatamente os mesmos dados**: mesma ordem de batches e mesmas augmentations. No código, os dois cenários **compartilhavam o mesmo `DataLoader`**, cujo gerador aleatório avança a cada época. O Scratch "herdava" o gerador já usado pelo Transfer e via outra ordem e outras augmentations. A correção cria um `DataLoader` novo, com a mesma seed, para cada cenário.

Arquivo alterado: [`src/services/experiment_runner.py`](../../src/services/experiment_runner.py). Para entender o bug, é preciso entender também [`src/repositories/dataset_repository.py`](../../src/repositories/dataset_repository.py), que não foi alterado.

---

## 1. Por que o pareamento importa (a estatística)

O estudo compara as médias com um **teste t pareado**. Para cada seed *i*, calcula-se a diferença:

```
dᵢ = F1_Transfer,i − F1_Scratch,i
t  = média(d) / (desvio(d) / √30)
```

O teste fica mais forte (t maior) quanto **menor o desvio das diferenças**. Pela estatística básica:

```
Var(d) = Var(X₁) + Var(X₂) − 2·Cov(X₁, X₂)
```

Se os dois cenários de uma mesma seed passam pelas mesmas condições aleatórias (mesma ordem dos dados, mesmas augmentations), o "azar" ou a "sorte" de uma seed afeta os dois parecido. Eles ficam **correlacionados** (Cov > 0), e esse ruído comum **se cancela** na subtração. Sobra na diferença só o efeito da técnica C, que é o que queremos medir.

Se cada cenário recebe dados sorteados de forma independente, a covariância vem dessa parte e o ruído **não** se cancela. O teste continua válido, mas é menos preciso, e o artigo afirma um controle experimental que não existia.

> É o mesmo princípio de um estudo clínico em que cada paciente testa os dois tratamentos: comparar o paciente com ele mesmo elimina a variação entre pessoas.

---

## 2. Como os dados chegam ao modelo: `dataset_repository.py`

### 2.1 As transformações (Técnica A)

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),                 # tamanho de entrada da ResNet-50
        transforms.RandomHorizontalFlip(p=0.5),        # espelha a imagem com 50% de chance
        transforms.RandomRotation(degrees=10),         # gira de −10° a +10°, sorteado
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # varia brilho/contraste em ±10%, sorteado
        transforms.ToTensor(),                         # PIL → tensor [3, 224, 224] com valores de 0 a 1
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_test_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
```

- O **treino** usa *data augmentation*: cada vez que uma imagem é carregada, ela sai um pouco diferente (espelhada ou não, levemente girada, mais clara ou mais escura). O modelo vê "variações" das 4077 radiografias e generaliza melhor. Essas transformações são **aleatórias**, e é por isso que o pareamento precisa controlar a aleatoriedade.
- O **teste** não tem nada aleatório: a mesma imagem sempre gera o mesmo tensor. Por isso o conjunto de teste é idêntico em todas as 60 execuções.
- `Normalize` subtrai a média e divide pelo desvio de cada canal usando as estatísticas do **ImageNet**, a normalização com que os pesos pré-treinados foram treinados. Aplicada aos dois cenários, ela é parte da Técnica A fixa.

### 2.2 O `Dataset`: como uma imagem é encontrada e carregada

```python
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = self.root_dir / class_name
            if class_folder.exists():
                for img_path in sorted(class_folder.glob("*.jpg")):
                    self.samples.append((img_path, class_idx))
        if not self.samples:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

- O construtor monta uma lista `(caminho, rótulo)` lendo as pastas `data/train/NORMAL` e `data/train/PNEUMONIA`. O `sorted(...)` garante que **a posição de cada imagem na lista é sempre a mesma**, em qualquer computador. Sem ele, a ordem dependeria do sistema de arquivos, e "a imagem nº 17" poderia ser diferente na máquina de cada um.
- `__len__` e `__getitem__` são o "contrato" de um `Dataset` do PyTorch: quantos itens existem e como obter o item *idx*.
- `__getitem__` só abre o arquivo **quando a imagem é pedida** (carregamento preguiçoso). O `.convert("RGB")` transforma o raio-X, que é em tons de cinza, em 3 canais, porque a ResNet espera imagens coloridas.
- **A augmentation acontece aqui, dentro do `transform`**, e usa o gerador aleatório do `torch` do processo onde o `__getitem__` roda (veja 2.4).

### 2.3 O `DataLoader` de treino

```python
def get_train_loader(self, batch_size=64, seed=1001, shuffle=True) -> DataLoader:
    dataset = self.get_train_dataset(augment=True)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=4,
    )
```

- O `DataLoader` junta itens do `Dataset` em batches de 64.
- `shuffle=True`: a cada época, a ordem das imagens é embaralhada.
- `generator=...`: em vez de usar o gerador aleatório global, o loader usa **um gerador próprio**, iniciado com a seed. É ele que torna o embaralhamento e as augmentations reproduzíveis.
- `num_workers=4`: quatro **processos auxiliares** carregam e transformam as imagens em paralelo enquanto a GPU treina. Abrir JPEG, redimensionar e girar é trabalho de CPU, e sem workers a GPU ficaria esperando.

### 2.4 Onde exatamente a aleatoriedade é sorteada

Conferido no código-fonte do PyTorch instalado (`torch/utils/data/`). **Cada vez que uma nova época começa**, isto é, cada vez que o `for x, y in train_loader:` cria um novo iterador, duas coisas são sorteadas **do gerador do loader**:

1. **A semente base dos workers** (`dataloader.py`, ~linha 697):
   ```python
   self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
   ```
   Cada worker *k* inicia seu próprio gerador com `base_seed + k` (`_utils/worker.py`, ~linha 258: `seed = base_seed + worker_id; random.seed(seed); torch.manual_seed(seed)`). São esses geradores que decidem os flips, as rotações e o brilho de cada imagem.
2. **A ordem das imagens** (`sampler.py`, `RandomSampler`): `torch.randperm(4077, generator=self.generator)`, uma permutação aleatória das posições 0..4076.

Os batches são distribuídos aos workers em rodízio (batch 0 → worker 0, batch 1 → worker 1, …), o que também é determinístico.

**Conclusão:** a sequência de dados de uma época depende **só do estado do gerador do loader no início da época**. Gerador no mesmo estado = mesma ordem e mesmas augmentations. E o gerador **avança** a cada época, porque cada sorteio consome números dele.

---

## 3. O problema

### 3.1 O código antigo (`run_single_seed`)

```python
train_loader = self.repo.get_train_loader(batch_size=self.batch_size, seed=seed, shuffle=True)

# Cenário 1: Transfer Learning
metrics_s1, _ = self.train_and_evaluate(pretrained=True,  seed=seed, train_loader=train_loader, ...)
# Cenário 2: Scratch
metrics_s2, _ = self.train_and_evaluate(pretrained=False, seed=seed, train_loader=train_loader, ...)
```

Um único loader, ou seja, **um único gerador**, era usado pelos dois cenários, um depois do outro.

### 3.2 Linha do tempo do gerador (2 épocas)

| Momento | Sorteios consumidos do gerador | Quem usa |
|---|---|---|
| Transfer, época 0 | sorteio nº 1 (base_seed + permutação) | Transfer |
| Transfer, época 1 | sorteio nº 2 | Transfer |
| Scratch, época 0 | sorteio nº **3** | Scratch ← deveria ser o nº 1 |
| Scratch, época 1 | sorteio nº **4** | Scratch ← deveria ser o nº 2 |

O Scratch continuava de onde o Transfer parou. Na "mesma seed", os dois cenários viam **ordens e augmentations diferentes**.

### 3.3 Prova

Um script de teste pegou os 2 primeiros batches de cada época e comparou uma "impressão digital" de cada batch (os 8 primeiros rótulos + a soma de todos os pixels, que muda com qualquer flip, rotação ou ajuste de brilho):

```
== ANTES (loader reaproveitado)
  época 0: batches idênticos nos 2 cenários? False
     cenário 1 batch 0: rótulos=(1, 0, 1, 1, 1, 0, 0, 1) soma_pixels=432629.2
     cenário 2 batch 0: rótulos=(1, 1, 0, 1, 1, 1, 1, 1) soma_pixels=1412506.5
  época 1: batches idênticos nos 2 cenários? False
     cenário 1 batch 0: rótulos=(0, 1, 1, 1, 1, 1, 1, 0) soma_pixels=-241623.4
     cenário 2 batch 0: rótulos=(0, 1, 1, 1, 1, 0, 0, 0) soma_pixels=1381114.8
```

> A soma pode ser negativa porque, depois do `Normalize`, os pixels ficam centrados em torno de zero.

---

## 4. A correção

```python
# Pareamento: cada cenário recebe um DataLoader novo com a mesma semente. Reaproveitar
# um único loader faria o cenário 2 herdar o gerador já avançado pelo cenário 1, e ele
# veria outra ordem de batches e outras augmentations.
def new_train_loader():
    return self.repo.get_train_loader(batch_size=self.batch_size, seed=seed, shuffle=True)

metrics_s1, _ = self.train_and_evaluate(pretrained=True,  seed=seed, train_loader=new_train_loader(), ...)
metrics_s2, _ = self.train_and_evaluate(pretrained=False, seed=seed, train_loader=new_train_loader(), ...)
```

- `new_train_loader` é uma **função definida dentro do método** (uma *closure*): ela "enxerga" as variáveis `self` e `seed` do método onde foi criada. Cada chamada constrói um loader novo, com um `torch.Generator().manual_seed(seed)` novo, portanto no **mesmo estado inicial**.
- Os dois cenários passam a consumir os sorteios nº 1 e nº 2, exatamente os mesmos.
- **Custo:** recriar o loader é barato. Só refaz a lista de caminhos dos arquivos (milissegundos); as imagens são lidas sob demanda durante o treino.
- O comentário fica no código de propósito: um "reaproveitar o loader para economizar" parece uma otimização inocente e traria o bug de volta.

> **Alternativa descartada:** reiniciar o gerador do loader existente (`train_loader.generator.manual_seed(seed)`) entre os cenários. Funciona, mas depende de detalhes internos do `DataLoader` e é fácil de esquecer em outro ponto do código. Um loader novo por cenário é mais explícito.

---

## 5. Verificação

### 5.1 Mesmo teste da seção 3.3, com a lógica nova

```
== DEPOIS (loader novo por cenário)
  época 0: batches idênticos nos 2 cenários? True
     cenário 1 batch 0: rótulos=(1, 0, 1, 1, 1, 0, 0, 1) soma_pixels=432629.2
     cenário 2 batch 0: rótulos=(1, 0, 1, 1, 1, 0, 0, 1) soma_pixels=432629.2
  época 1: batches idênticos nos 2 cenários? True
     cenário 1 batch 0: rótulos=(0, 1, 1, 1, 1, 1, 1, 0) soma_pixels=-241623.4
     cenário 2 batch 0: rótulos=(0, 1, 1, 1, 1, 1, 1, 0) soma_pixels=-241623.4
```

Rótulos **e** soma de pixels idênticos: a ordem e as augmentations batem. Repare também que o cenário 1 é igual ao de antes. Ele sempre usou o gerador "zerado"; quem mudou foi o cenário 2.

### 5.2 Treino real (seed 1001, 2 épocas, com as correções 3 e 4)

| Seed 1001 | Transfer | Scratch (antes) | Scratch (depois) |
|---|---|---|---|
| F1-macro | 0,9464 (idêntico antes e depois) | 0,8777 | 0,8138 |
| Acurácia | 0,9570 | 0,8918 | 0,8643 |

O Transfer saiu **idêntico** nas duas execuções, coerente com a seção 5.1: os dados dele não mudaram. O Scratch mudou porque recebeu outros dados (esta correção) e outro learning rate (correção 3). Uma seed só não diz se ele melhora ou piora em média; isso exige as 30.

---

## 6. O que continua não idêntico (e por quê)

- **Pesos iniciais:** um cenário começa do ImageNet e o outro de pesos aleatórios. Essa é **a variável sendo testada**, então é diferente de propósito.
- **Não determinismo da GPU:** algumas operações de convolução somam números em paralelo, numa ordem que pode variar, e somas em ponto flutuante não são exatamente associativas. Isso pode gerar diferenças minúsculas entre execuções. No teste acima, o Transfer reproduziu exatamente, mas isso não é garantido em toda GPU ou versão de driver.
- **`num_workers` faz parte do experimento:** como cada worker semeia seu gerador com `base_seed + id`, mudar o número de workers muda quais augmentations cada imagem recebe. Para reproduzir exatamente, todos devem usar `num_workers=4`.

---

## 7. Pendências

1. **Refazer as 30 runs**, junto com as outras correções, para que o artigo possa afirmar o pareamento com verdade.
2. **Possível otimização futura:** no Windows, os 4 workers são recriados a cada época (`persistent_workers=False`), e cada um importa o PyTorch de novo. Isso explica parte do tempo extra por seed. `persistent_workers=True` aceleraria, mas muda **quando** os workers são semeados. Se for aplicado, precisa ser verificado de novo com o teste da seção 5.1.

## 8. Lição geral

> **Objetos com estado aleatório (geradores, iteradores, loaders) não devem ser compartilhados entre condições que deveriam ser idênticas.** Reaproveitar parece economia, mas carrega o estado de uma condição para a outra. Para garantir que duas execuções vejam a mesma aleatoriedade, crie o recurso do zero com a mesma seed para cada uma e **teste a igualdade**: compare os batches, não só as configurações.
