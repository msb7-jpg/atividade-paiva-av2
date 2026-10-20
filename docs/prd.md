Aqui está o mapeamento completo e estruturado da solução, seguindo rigorosamente o enunciado da 2ª Avaliação, o exemplo de metodologia aprovado na TV e o conteúdo dos slides da disciplina.

---

## 1. Análise do Objetivo e do Problema

O objetivo da avaliação não é apenas criar um modelo que funcione bem, mas realizar um **experimento científico controlado** para provar estatisticamente que uma escolha técnica específica é superior a outra.

### O que precisamos fazer exatamente (Requisitos do Enunciado):

1. **Definir um Caso de Uso (Problema):** Escolher uma tarefa de visão computacional com um conjunto de dados real (ex.: classificar imagens de folhas de plantas ou objetos em 3 classes).


2. **Isolar 3 Técnicas da Disciplina:** Selecionar três métodos ensinados nas aulas para compor o pipeline do modelo.


3. **Montar 2 Cenários (Controle vs. Experimental):** Manter a Técnica A e a Técnica B fixas em ambos os lados, e alterar apenas a Técnica C:


* **Cenário 1 (Principal):** Técnica A + Técnica B + **Técnica C**.


* **Cenário 2 (Contrafactual / Alternativo):** Técnica A + Técnica B + **Técnica C (Alternativa)**.




4. **Executar 30 Simulações Pareadas:** Rodar o treinamento/avaliação 30 vezes para cada cenário, variando a semente aleatória (*seeds* de 1001 a 1030), garantindo que em cada par $(i)$ ambos os modelos sejam avaliados sob as mesmas condições.


5. **Inferência Estatística (Hipótese Nula vs. Alternativa):** Aplicar um **Teste $t$ pareado bilateral** sobre os 30 pares de resultados (F1-Macro e Acurácia) para calcular a diferença média, o intervalo de confiança de 95% e o $p$-valor:


* **Hipótese Nula ($H_0$):** As duas abordagens têm eficácia igual (a diferença média de desempenho é zero).


* **Hipótese Alternativa ($H_1$):** Uma das abordagens é estatisticamente mais eficaz que a outra ($p < 0.05$).




6. **Gerar os 3 Entregáveis:** Código funcional + Apresentação de slides + Relatório em formato de artigo **IEEE de coluna dupla** contendo os gráficos comparativos.



---

## 2. Escolha Estratégica das Técnicas (Baseada nos Slides da Aula)

Para que o código rode as 60 execuções (30 pares) em **menos de 2 minutos no Google Colab** e utilize exatamente o que o Prof. Rodrigo Paiva destacou nos slides, a melhor abordagem é usar **Foundation Models / CLIP com Transfer Learning**.

### Caso de Uso Escolhido:

* **Dataset:** *Beans Dataset* (o mesmo do slide da TV: 1.034 imagens de treino, 133 de validação e 128 de teste, divididas em 3 classes: *Mancha Angular*, *Ferrugem* e *Saudável*) - ou qualquer dataset de 3 classes do Hugging Face/Torchvision.



### Mapeamento das 3 Técnicas e Como Elas Solucionam o Problema:

| Componente | Técnica Escolhida | Onde está nos Slides | Papel na Solução do Problema |
| --- | --- | --- | --- |
| **Técnica A (Fixa)** | **Extração de Features com ViT / CLIP** (`clip-vit-base-patch32`) | Slide 2 (*Transformers / ViT*), Slide 8 (*ViT*), Slide 12 (*CLIP*) e Slide 29 (*Código Colab*).

 | Transforma cada imagem bruta (pixels) em um vetor numérico rico de 512 dimensões (*embedding*) usando pesos pré-treinados em 400 milhões de pares da web.

 |
| **Técnica B (Fixa)** | **Prevenção a Overfitting com Dropout (0.2) + Regularização Ridge ($L_2$)** | Slide da TV (*Modelo e treinamento: Dropout 0.2 e Regularização Ridge L2*).

 | Na cabeça de classificação (camada densa final), desliga aleatoriamente 20% dos neurônios no treino e penaliza pesos altos, evitando que o modelo decore os dados de treino.

 |
| **Técnica C (Cenário 1)** | **Fine-Tuning Supervisionado (Linear Probe + Cabeça Treinada com Adam)** | Slide 2 (*Transfer learning: congelar ou ajustar; linear probe*), Slide 13 e Slide 30 (*Siglário*).

 | Mantém o backbone congelado e treina uma cabeça classificadora (Pooling/Dropout + Camada Densa com Softmax de 3 saídas) usando os rótulos do dataset de treino.

 |
| **Técnica C Alternativa (Cenário 2)** | **Classificação Zero-Shot Multimodal via Similaridade de Cosseno (ou Inicialização Aleatória)** | Slide 2 (*Few-shot / distância*), Slide 13 (*Zero-shot*) e Slide 29 (ou Slide da TV: *Pesos Aleatórios*).

 | **Opção C1 (Fiel à TV):** Treinar a mesma arquitetura mas com **pesos aleatórios** (sem pré-treino).

<br>

<br>**Opção C2 (Fiel ao PDF novo):** Classificar ou inicializar a cabeça comparando os vetores de texto das classes via similaridade de cosseno.

 |

> **Recomendação prática de ouro:** Para ficar **100% blindado** tanto pelo slide da TV quanto pelo PDF do professor, usamos a estrutura idêntica à da TV trocando apenas o modelo antigo (MobileNetV2) pelo modelo da aula dele (**ViT - Vision Transformer**) ou mantendo a comparação de **Transfer Learning vs. Pesos Aleatórios** (ou **Linear Probe vs. Fine-Tuning**).
> 
> 

---

## 3. Detalhamento da Arquitetura Comparativa (O que vai para o Slide e Relatório)

Seguindo exatamente o fluxograma da foto da TV (`1000466654.jpg`), a sua metodologia fica desenhada assim:

### Passo 1: Dados e Preparação (Técnica A)

* **Conjunto de Dados:** *Beans Dataset* (3 classes balanceadas).


* **Pré-processamento e Redimensionamento:** Redimensionamento das imagens para $224 \times 224$ pixels (padrão do ViT/ResNet) e normalização dos canais RGB, reduzindo o custo computacional.


* **Data Augmentation no Treino:** Espelhamento horizontal aleatório e variação de brilho.



### Passo 2: Divisão nos 2 Cenários (Técnica C vs. Técnica C Alternativa)

* **Cenário 1 (Solução Principal - Com Transfer Learning):** Backbone pré-treinado no ImageNet/CLIP congelado (ou ajustado) extraindo representações visuais consolidadas.


* **Cenário 2 (Solução Contrafactual - Sem Transfer Learning / Alternativa):** Mesma arquitetura, porém com **inicialização de pesos aleatórios** (ou sem ajuste fino), forçando a rede a aprender apenas com as ~1.000 imagens do dataset.



### Passo 3: Cabeça Comum e Regularização (Técnica B)

* Ambos os cenários passam pela mesma estrutura final:
* **Dropout de $0.2$** + **Regularização $L_2$ (Weight Decay = $10^{-4}$)** para prevenção de overfitting.


* Camada Densa final com **3 saídas e ativação Softmax** (uma probabilidade para cada classe).


* Otimizador **Adam** (taxa de aprendizado $10^{-4}$) e lote (*batch size*) de 32 imagens.





---

## 4. O Protocolo de Controle e Avaliação Estatística

Para cumprir o requisito das **30 simulações e inferência de hipótese**:

1. **Controle por Sementes (*Seeds* 1001 a 1030):**
* Criamos um loop `for seed in range(1001, 1031):`.


* Em cada `seed`, fixamos a aleatoriedade do PyTorch/NumPy (`torch.manual_seed(seed)`).
* Treinamos o **Cenário 1** e avaliamos no conjunto de teste (128 imagens), salvando o **F1-Macro** e a **Acurácia**.


* Treinamos o **Cenário 2** com a *mesma seed* e avaliamos no *mesmo conjunto de teste*, salvando o **F1-Macro** e a **Acurácia**.




2. **Teste Estatístico (Teste $t$ Pareado Bilateral):**
* Ao final das 60 execuções, teremos um vetor $X$ com 30 valores do Cenário 1 e um vetor $Y$ com 30 valores do Cenário 2.


* Calculamos o vetor de diferenças pareadas $d_i = X_i - Y_i$ para $i = 1, \dots, 30$.


* Aplicamos a estatística $t$:

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$



onde $\bar{d}$ é a média das diferenças, $s_d$ é o desvio padrão das diferenças e $n = 30$.
* Se o **$p$-valor for menor que $0.05$**, rejeitamos a **Hipótese Nula ($H_0$)** e concluímos que a **Hipótese Alternativa ($H_1$)** é verdadeira (o Cenário 1 é comprovadamente mais eficaz).




3. **Explicabilidade Visual (XAI - Opcional de alto valor):**
* Assim como no slide da TV (que usou Grad-CAM) e no Slide 2 e 10 do PDF (mapas de atenção/saliência), geramos o mapa visual em 3 imagens fixas de teste para mostrar onde o modelo pré-treinado foca (nas lesões da folha) versus onde o modelo alternativo foca (espalhado pelo fundo).





---

Quer que eu já gere o **notebook completo em Python (pronto para colar no Colab)** com esse pipeline exato já gerando os gráficos e a tabela estatística para o relatório IEEE?
