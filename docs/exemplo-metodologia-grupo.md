# 2ª Avaliação - Inteligência Artificial (Prof. Rodrigo Paiva)

## 2. Transcrição Fiel da Imagem 1: Slide de Metodologia na TV (`1000466654.jpg`)

### Texto Literal na Coluna da Esquerda (Tópicos):

> **Metodologia**
> 
> - **Dados e Divisão das Imagens**
> 
> 
> - Três classes: Mancha Angular, Ferrugem e Saudável.
> - Fotografias 500x500 pixels, RGB.
> 
> 
> - **Preparação Imagens**
> 
> 
> - Conversão de RGB para HSV e criação da Máscara.
> - Preenchimento do contorno e substituição do fundo.
> - Redimensionar imagens, reduzindo custo computacional.
> 
> 
> - **Modelo e treinamento**
> 
> 
> - MobileNetV2 como base.
> - Cabeça possui 3 saídas, uma para cada classe a ser prevista.
> - Prevenção a overfitting: Dropout 0.2 e Regularização Ridge (L2) .
> - *Data Augmentation* com espelhamento horizontal e variação de brilho.
> - Treinamento 100 épocas, *batch* de 32 imagens.
> 
> 
> - **Controle**
> 
> 
> - 30 simulações pareadas, totalizando 60 treinamentos.
> - Mesmo conjunto de teste em todas as execuções.
> 
> 
> - **Avaliação**
> 
> 
> - F1 Macro, as três classes tem o mesmo peso.
> - Uso Teste T pareado, entender diferença média de desempenho para as duas inicializações.
> - Grad-CAM, identificar as regiões associadas à classe prevista.
> 
> 

### Texto Literal na Coluna da Direita (Fluxograma de Blocos):

*(Lido de cima para baixo, seguindo as setas)*

1. **Bloco 1 (Topo):**
`Conjunto Beans (3 classes)`


`treino 1.034 · validação 133 · teste 128`


$\downarrow$
2. **Bloco 2:**
`Segmentação de silhueta (HSV)`


`máscara HSV · fundo cinza 127 · 160×160 px`


$\swarrow \quad \searrow$ *(Divide-se em dois caminhos paralelos)*
3. **Bloco 3A (Esquerda - Azul):**
`Solução principal`


`MobileNetV2 + pesos ImageNet`


**Bloco 3B (Direita - Bege):**
`Solução contrafactual`


`MobileNetV2 + pesos aleatórios`


$\searrow \quad \swarrow$ *(Ambos convergem para o Bloco 4)*
4. **Bloco 4:**
`Cabeça comum às duas soluções`


`pooling global · dropout 0,2 · Dense 3 (softmax)`


$\downarrow$
5. **Bloco 5:**
`Treino fim a fim (todas as camadas treináveis)`


`100 épocas · Adam (0^{-4}$) · lote 32`


`aumento: espelhamento e brilho (±0,05 × 255)`


`pesos da época de menor perda de validação`


$\downarrow \quad \searrow$ *(Ramifica para avaliação quantitativa à esquerda e qualitativa à direita)*
6. **Caminho Esquerdo (Quantitativo):**
* **Bloco 6A:** `Avaliação no teste (128 imagens)` / `F1 macro · acurácia`

* $\downarrow$
* **Bloco 7A:** `30 pares (seeds 1001-1030)` / `60 treinamentos`

* $\downarrow$
* **Bloco 8A:** `Teste t pareado (bilateral)` / `diferença média, IC 95%, valor-p`



7. **Caminho Direito (Qualitativo - XAI):**
* **Bloco 6B:** `Grad-CAM` / `alvo: classe prevista` / `3 imagens fixas` / `× 2 soluções × 30 seeds` / `(análise qualitativa)`




---

## 3. O que se Extrai da Lógica dessa Metodologia (Por que esse grupo fez assim?)

Analisando os detalhes matemáticos e de engenharia nos blocos do fluxograma, extrai-se exatamente como esse grupo traduziu o enunciado do professor em código:

1. **Mapeamento Exato das 3 Técnicas da Regra:**
* **Técnica A (Pré-processamento/Visão Clássica):** Segmentação de silhueta por cor HSV (eles removeram a terra/fundo das fotos das folhas e pintaram o fundo de cinza neutro `RGB 127` para a rede não trapacear olhando para o chão em vez de olhar para a folha, reduzindo a imagem de $500 \times 500$ para $160 \times 160$ para treinar rápido).


* **Técnica B (Arquitetura e Regularização):** CNN MobileNetV2 com Global Average Pooling, Dropout de $20\%$ e regularização $L_2$ (Ridge) + *Data Augmentation* (espelhamento e brilho $\pm 5\%$).


* **Técnica C vs. Técnica C (Alternativa):** *Transfer Learning* (pesos pré-treinados na ImageNet) **VS.** Inicialização Aleatória (treinar a MobileNetV2 do zero absoluto).




2. **Por que "Simulações Pareadas" com `seeds 1001-1030`?**
* Parear significa que, na simulação nº 1 (`seed = 1001`), a sequência exata de sorteio dos lotes (*batches*) de 32 imagens e os sorteios de espelhamento/brilho do *Data Augmentation* são controlados pela semente `1001`.


* Tanto a **Solução Principal** quanto a **Solução Contrafactual** rodam sob a `seed 1001`. Isso elimina o ruído externo: ambas enfrentam a exata mesma ordem de dados e transformações, mudando única e exclusivamente os pesos iniciais.




3. **Critério de Parada (*Model Checkpointing*):**
* No Bloco 5 está escrito: `pesos da época de menor perda de validação`. Em vez de pegar cegamente o modelo da época 100 (que poderia estar sofrendo *overfitting*), eles monitoraram a perda (*loss*) nas 133 imagens de validação a cada época e salvaram o melhor *checkpoint* para avaliar nas 128 imagens de teste.




4. **Uso Inteligente do Grad-CAM (XAI):**
* Em vez de gerar Grad-CAM para todas as imagens (o que seria inviável), eles congelaram **3 imagens fixas** (1 de Mancha Angular, 1 de Ferrugem e 1 Saudável) e geraram os mapas de calor para comparar visualmente no relatório se o modelo com ImageNet olha mais para a mancha da doença do que o modelo com pesos aleatórios.
