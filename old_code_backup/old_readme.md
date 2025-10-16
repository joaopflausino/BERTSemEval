# **Relatório de Classificação de Sentimentos usando o Conjunto de Dados SemEval (2013–2017)**

## **1. Visão Geral**

Este projeto envolve o treinamento de um modelo de classificação de sentimentos usando conjuntos de dados SemEval de 2013 a 2016 e a avaliação de seu desempenho no conjunto de dados de 2017. O modelo foi construído usando uma arquitetura baseada em BERT e treinado por três períodos.

Links para os repositórios utilizados como base


[Twitter Sentiment Analysis with CNNs and LSTMs](https://paperswithcode.com/paper/bb_twtr-at-semeval-2017-task-4-twitter)
[International Workshop on Semantic Evaluation](https://semeval.github.io/)
[Dataset](https://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools)
[leelaylay /TweetSemEval](https://github.com/edilsonacjr/semeval2017)
[edilsonacjr /semeval2017](https://github.com/edilsonacjr/semeval2017)

---

## **2. Configuração do Conjunto de Dados**

* **Dados de Treinamento**: Conjuntos de dados SemEval de 2013 a 2016 (`dataset/train/`)
* **Dados de Validação/Teste**: SemEval 2017, Subtarefa A, Inglês (`dataset/test/SemEval2017-task4-test.subtask-A.english.txt`)
* **Máximo de Tokens por Amostra**: 128

---

## **3. Arquitetura do Modelo**

* **Modelo Base**: BERT (`bert-base-uncased`)
* **Dimensão da Saída do BERT**: 768
* **Componentes da Arquitetura**:

  * **Dropout Inicial**: 0.5
  * **Camada Oculta**:

    * Linear: 768 → 384 (redução pela metade da dimensão do BERT)
    * Ativação: ReLU
    * Dropout: 0.5
  * **Camada de Classificação Final**:

    * Linear: 384 → 3 (positivo, negativo, neutro)

---

## **4. Configuração de Treinamento**

* **Épocas**: 3
* **Tamanho do Lote**: 16 (amostrador de balde)
* **Otimizador**: AdamW (taxa de aprendizado = 2e-5, decaimento de peso = 0,01)
* **Agendador**: Triangular Inclinado
* **Paciência**: 2
* **Dispositivo**: GPU com CUDA

---

## **5. Métricas de Treinamento e Validação**

| Métrica | Epoch 0 | Epoch 1 | Epoch 2 |
| ----------------------- | ---------- | ------- | ---------- |
| **Precisão do Treinamento** | 66,72% | 79,06% | **87,26%** |
| **F1 de Treinamento** | 66,59% | 79,07% | **87,27%** |
| **Perda de Treinamento** | 0,7178 | 0,4973 | **0,3307** |
| **Precisão da Validação** | **70,56%** | 70,28% | 68,99% |
| **Validação F1** | **70,50%** | 70,24% | 69,00% |
| **Perda de Validação** | **0,6620** | 0,6904 | 0,8605 |

* **Melhor Época**: **Epoch 0**
* **Duração do Treinamento**: 31 minutos e 18 segundos
* **Uso Máximo de Memória da CPU**: 1288 MB