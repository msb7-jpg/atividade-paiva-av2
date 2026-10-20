# 2ª Avaliação - Inteligência Artificial (Prof. Rodrigo Paiva)

## 1. Transcrição Fiel da Imagem 2: Instruções da Atividade (`1000466655.jpg`)

### Texto Literal na Tela:

> **2a AVALIAÇÃO**
> 
> **Estudo Comparativo:**
> 
> - Selecionar obrigatoriamente 03(três) técnicas abordadas na sala de aula;
> - Comparar a performance em 02(dois) cenários: técnica A + técnica B + técnica C VS técnica A + técnica B + técnica C(Alternativa);
> - Realizar 30 simulações e inferir qual das duas abordagens é mais eficaz: Hipótese Nula ou Hipótese Alternativa?
> - Entregar código com o caso de uso + apresentação + relatório em formato IEEE coluna dupla com os gráficos;
> 
> 
> **OBSERVAÇÃO:**
> 
> 
> **Datas:**
> 
> 
> **grupo 01 e 02 - 20/Out**
> 
> 
> **grupo 03 e 04 - 22/Out**
> 
> 
> **grupo 05 e 06 - 27/Out**
> 
> 
> **grupo 07 e 08 - 29/Out**
> 
> 
> **grupo 09 e 10 - 10/Nov**
> 
> 
> **grupo 11 e 12 - 12/Nov**
> 

### Lógica Extraída das Instruções:

1. **Isolamento de Variável Experimental (*Ablation / A/B Test*):** Ao exigir `A + B + C` versus `A + B + C(Alternativa)`, o professor está exigindo o princípio científico de *ceteris paribus* (tudo o mais constante). Você não pode mudar a preparação dos dados e o modelo ao mesmo tempo; se mudar duas coisas juntas, não saberá qual delas causou a melhora na nota. A e B funcionam como **constantes de controle**, enquanto C vs. C(Alternativa) é a **variável independente** do estudo.


2. **Amostragem Estatística ($N = 30$):** Em Machine Learning, rodar um treino apenas uma vez pode dar um resultado bom ou ruim por pura sorte da inicialização aleatória dos pesos ou do embaralhamento dos lotes (*batches*). Exigir 30 simulações garante que a distribuição da média das métricas se aproxime de uma distribuição normal (pelo Teorema Central do Limite, onde $N \ge 30$ é o limiar clássico), permitindo aplicar testes paramétricos como o Teste $t$.


3. **Decisão Baseada em Hipóteses:** A resposta para "qual abordagem é mais eficaz" não pode ser apenas "a média de C foi 85% e a de C_alt foi 82%". É obrigatório provar via teste de hipótese que essa diferença de 3% não ocorreu ao acaso:


* **Hipótese Nula ($H_0$):** $\mu_C - \mu_{C_{\text{alt}}} = 0$ (As duas técnicas têm eficácia estatisticamente idêntica; qualquer variação observada é ruído aleatório).
* **Hipótese Alternativa ($H_1$):** $\mu_C - \mu_{C_{\text{alt}}} \neq 0$ (Existe uma diferença real e estatisticamente significativa de performance entre usar C e usar C Alternativa).



---
