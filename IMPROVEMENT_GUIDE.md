# Guia de Melhorias para Validation Loss Alta

Este guia fornece soluções práticas para melhorar os resultados do modelo quando a validation loss está alta.

## 🔍 Passo 1: Diagnosticar o Problema

Primeiro, identifique exatamente qual é o problema:

```bash
# Se você já treinou um modelo
python diagnose_model.py --experiment experiments/bert_base

# Se você quer analisar seus dados
python diagnose_model.py --train-data dataset/processed/bert_train.csv --val-data dataset/processed/bert_validation.csv

# Ambos
python diagnose_model.py --experiment experiments/bert_base --train-data dataset/processed/bert_train.csv
```

O script irá diagnosticar se você tem:
- **OVERFITTING**: Training loss baixa, validation loss alta
- **UNDERFITTING**: Ambas as losses altas
- **DESBALANCEAMENTO**: Classes desbalanceadas nos dados

## 📊 Cenário 1: OVERFITTING

**Sintomas:**
- Training loss: 0.2-0.4
- Validation loss: > 0.8
- Gap entre training e validation > 0.4

**Solução: Use a configuração anti-overfitting**

```bash
python train_model.py --config configs/bert_anti_overfit.yaml
```

**O que essa config faz:**
- ✅ **Dropout alto (0.4)**: Previne memorização
- ✅ **Weight decay forte (0.1)**: Penaliza pesos grandes
- ✅ **Learning rate baixo (1e-5)**: Aprendizado mais cauteloso
- ✅ **Freeze layers (6)**: Congela camadas iniciais do BERT
- ✅ **Focal Loss**: Foca em exemplos difíceis
- ✅ **Early stopping (patience=3)**: Para quando começar a overfitar
- ✅ **Data augmentation (30%)**: Aumenta diversidade dos dados

**Ajustes adicionais se ainda não melhorar:**

1. **Congele mais camadas**:
   ```yaml
   model:
     freeze_layers: 9  # Congela quase todo o BERT
   ```

2. **Aumente dropout ainda mais**:
   ```yaml
   model:
     dropout_prob: 0.5  # Máximo recomendado
   ```

3. **Reduza learning rate**:
   ```yaml
   training:
     learning_rate: 5e-6  # Extremamente baixo
   ```

## 📊 Cenário 2: UNDERFITTING

**Sintomas:**
- Training loss: > 0.8
- Validation loss: > 0.9
- Modelo não está aprendendo adequadamente

**Solução: Use a configuração anti-underfitting**

```bash
python train_model.py --config configs/bert_anti_underfit.yaml
```

**O que essa config faz:**
- ✅ **Dropout baixo (0.05)**: Permite que o modelo aprenda mais
- ✅ **Sem freeze**: Todas as camadas treinadas
- ✅ **Learning rate alto (5e-5)**: Aprende mais rápido
- ✅ **Batch size maior (32)**: Gradientes mais estáveis
- ✅ **Mais épocas (20)**: Mais tempo para aprender
- ✅ **Weight decay mínimo (0.001)**: Menos penalização

**Ajustes adicionais:**

1. **Verifique seus dados**:
   ```bash
   python diagnose_model.py --train-data dataset/processed/bert_train.csv
   ```
   - Textos muito curtos ou muito longos?
   - Dados mal preprocessados?
   - Ruído excessivo?

2. **Aumente max_length se textos são longos**:
   ```yaml
   data:
     max_length: 256  # Ou até 512
   ```

3. **Experimente learning rate ainda maior**:
   ```yaml
   training:
     learning_rate: 8e-5  # Limite superior recomendado
   ```

## 📊 Cenário 3: Configuração Balanceada

**Se você não sabe qual é o problema ou quer um ponto de partida sólido:**

```bash
python train_model.py --config configs/bert_balanced.yaml
```

**O que essa config faz:**
- ✅ **Parâmetros balanceados** seguindo best practices
- ✅ **Label smoothing**: Previne overconfidence
- ✅ **Freeze moderado (3 camadas)**: Equilíbrio entre flexibilidade e overfitting
- ✅ **Augmentation leve (20%)**: Melhora generalização sem dificultar treino

## 🔧 Melhorias Avançadas Disponíveis

### 1. Loss Functions Avançadas

Você pode modificar o tipo de loss no config:

```yaml
training:
  # Focal Loss - melhor para classes desbalanceadas
  loss_type: "focal"
  focal_gamma: 2.0  # Maior = mais foco em exemplos difíceis

  # Label Smoothing - previne overconfidence
  loss_type: "label_smoothing"
  label_smoothing: 0.1  # 0.1 a 0.2 é recomendado

  # Symmetric Cross Entropy - robusto a ruído nos labels
  loss_type: "symmetric"

  # Combined - usa múltiplas losses
  loss_type: "combined"
```

### 2. Layer Freezing

Congele camadas do BERT para reduzir overfitting:

```yaml
model:
  freeze_layers: 6  # Congela primeiras 6 de 12 camadas
  # 0 = nenhuma
  # 6 = metade
  # 9 = quase todas
  # null = apenas embeddings
```

### 3. Data Augmentation

Ative augmentation para melhorar generalização:

```yaml
training:
  use_augmentation: true
  augmentation_prob: 0.3  # 30% dos dados serão aumentados
```

Técnicas aplicadas:
- Synonym replacement
- Random insertion
- Random swap
- Random deletion

### 4. Class Weights

Para dados desbalanceados:

```yaml
training:
  use_class_weights: true
  class_weight_method: "balanced"  # Ou "inverse_freq"
```

### 5. Early Stopping

Ajuste para parar no momento certo:

```yaml
training:
  early_stopping:
    patience: 3      # Número de épocas sem melhoria
    min_delta: 0.001 # Melhoria mínima considerada
```

## 📈 Workflow Recomendado

### Passo a Passo:

1. **Diagnóstico inicial**:
   ```bash
   python diagnose_model.py --experiment experiments/bert_base --train-data dataset/processed/bert_train.csv
   ```

2. **Identifique o problema** baseado na saída do diagnóstico

3. **Escolha a configuração apropriada**:
   - Overfitting → `bert_anti_overfit.yaml`
   - Underfitting → `bert_anti_underfit.yaml`
   - Incerto → `bert_balanced.yaml`

4. **Treine o modelo**:
   ```bash
   python train_model.py --config configs/bert_anti_overfit.yaml
   ```

5. **Analise os resultados**:
   ```bash
   python diagnose_model.py --experiment experiments/bert_anti_overfit
   ```

6. **Itere**: Ajuste hiperparâmetros baseado nos resultados

## 🎯 Tabela de Referência Rápida

| Problema | Dropout | LR | Weight Decay | Freeze | Epochs | Loss Type |
|----------|---------|-------|--------------|--------|--------|-----------|
| **Overfitting severo** | 0.4-0.5 | 1e-5 | 0.05-0.1 | 6-9 | 10-15 | Focal |
| **Overfitting moderado** | 0.2-0.3 | 2e-5 | 0.02-0.05 | 3-6 | 10-12 | Label Smoothing |
| **Balanceado** | 0.15-0.2 | 2e-5 | 0.01 | 3 | 10-15 | Label Smoothing |
| **Underfitting** | 0.05-0.1 | 3e-5-5e-5 | 0.001-0.01 | 0 | 15-20 | Cross Entropy |

## 💡 Dicas Extras

### Se validation loss continua alta:

1. **Verifique preprocessamento**:
   - Remover URLs, menções, hashtags
   - Normalizar texto
   - Remover caracteres repetidos

2. **Analise distribuição de dados**:
   ```python
   # Verifique se train e validation têm distribuição similar
   python diagnose_model.py --train-data dataset/processed/bert_train.csv --val-data dataset/processed/bert_validation.csv
   ```

3. **Aumente max_length se textos são longos**:
   ```yaml
   data:
     max_length: 256  # Padrão é 128
   ```

4. **Use modelo maior se dataset é grande**:
   ```yaml
   model:
     name: "bert-large-uncased"  # Em vez de bert-base
   ```

5. **Combine múltiplas técnicas**:
   - Focal Loss + Data Augmentation + Layer Freezing
   - Label Smoothing + Class Weights + Early Stopping

## 📝 Exemplos de Uso Completo

### Exemplo 1: Resolver Overfitting Severo

```bash
# 1. Diagnóstico
python diagnose_model.py --experiment experiments/bert_base

# Output mostra: Training Loss: 0.25, Validation Loss: 1.1 (GAP: 0.85)

# 2. Treinar com config anti-overfitting
python train_model.py --config configs/bert_anti_overfit.yaml

# 3. Verificar melhoria
python diagnose_model.py --experiment experiments/bert_anti_overfit

# Output esperado: Gap reduzido para ~0.3-0.4
```

### Exemplo 2: Melhorar Underfitting

```bash
# 1. Diagnóstico mostra ambas losses altas (>0.8)

# 2. Verificar dados
python diagnose_model.py --train-data dataset/processed/bert_train.csv

# 3. Treinar com config anti-underfitting
python train_model.py --config configs/bert_anti_underfit.yaml

# 4. Se ainda não melhorar, aumentar learning rate manualmente
# Editar configs/bert_anti_underfit.yaml e trocar learning_rate para 8e-5
```

## 🚀 Melhores Práticas

1. **Sempre comece com diagnóstico**
2. **Mude um hiperparâmetro por vez** para entender o impacto
3. **Use early stopping** para evitar desperdício de tempo
4. **Salve múltiplos experimentos** e compare com `compare_models.py`
5. **Documente o que funciona** para seu dataset específico

## 📞 Troubleshooting

### "Training loss não diminui"
- Aumente learning rate (3e-5 a 5e-5)
- Reduza weight decay (0.001)
- Verifique se dados estão carregados corretamente

### "Validation loss oscila muito"
- Reduza learning rate
- Aumente batch size
- Use warmup maior (0.15-0.2)

### "Modelo converge muito rápido e overfita"
- Use early stopping com patience=3
- Aumente dropout
- Ative data augmentation

### "Classes desbalanceadas"
- Use `use_class_weights: true`
- Experimente Focal Loss
- Considere oversampling da classe minoritária

---

**Última atualização**: 2024-10-29
