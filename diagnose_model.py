#!/usr/bin/env python3
"""
Script de diagnóstico para identificar problemas de overfitting/underfitting
e analisar a qualidade dos dados
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent / "src"))

class ModelDiagnostics:
    def __init__(self, experiment_dir=None, data_path=None):
        self.experiment_dir = experiment_dir
        self.data_path = data_path

    def load_metrics(self):
        """Carrega métricas de treinamento"""
        if not self.experiment_dir:
            print("⚠️  Nenhum diretório de experimento fornecido")
            return None

        metrics_path = Path(self.experiment_dir) / "metrics.json"
        if not metrics_path.exists():
            print(f"❌ Arquivo de métricas não encontrado: {metrics_path}")
            return None

        with open(metrics_path, 'r') as f:
            return json.load(f)

    def diagnose_overfitting(self, metrics):
        """Diagnostica overfitting/underfitting baseado nas métricas"""
        if not metrics:
            return

        print("\n" + "="*60)
        print("🔍 DIAGNÓSTICO DE OVERFITTING/UNDERFITTING")
        print("="*60)

        epochs = []
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        # Extrai métricas de cada época
        for key in metrics.keys():
            if key.startswith("epoch_"):
                epoch_data = metrics[key]
                epochs.append(int(key.split("_")[1]))
                train_losses.append(epoch_data.get('training_loss', 0))
                val_losses.append(epoch_data.get('validation_loss', 0))
                train_accs.append(epoch_data.get('training_accuracy', 0))
                val_accs.append(epoch_data.get('validation_accuracy', 0))

        if not epochs:
            print("❌ Nenhuma métrica de época encontrada")
            return

        # Calcula gaps
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]

        loss_gap = final_val_loss - final_train_loss
        acc_gap = final_train_acc - final_val_acc

        print(f"\n📊 Métricas Finais (Época {epochs[-1] + 1}):")
        print(f"   Training Loss:    {final_train_loss:.4f}")
        print(f"   Validation Loss:  {final_val_loss:.4f}")
        print(f"   Loss Gap:         {loss_gap:.4f}")
        print(f"   Training Acc:     {final_train_acc:.4f}")
        print(f"   Validation Acc:   {final_val_acc:.4f}")
        print(f"   Accuracy Gap:     {acc_gap:.4f}")

        # Diagnóstico
        print(f"\n🎯 DIAGNÓSTICO:")

        if loss_gap > 0.5 and acc_gap > 0.1:
            print("   ⚠️  OVERFITTING SEVERO DETECTADO!")
            print("   O modelo memorizou os dados de treino mas não generaliza bem.")
            print("\n   ✅ SOLUÇÕES RECOMENDADAS:")
            print("   1. Aumentar dropout (tente 0.3-0.5)")
            print("   2. Aumentar weight_decay (tente 0.05-0.1)")
            print("   3. Reduzir learning rate (tente 1e-5 a 2e-5)")
            print("   4. Usar data augmentation")
            print("   5. Adicionar early stopping com patience=3")
            print("   6. Congelar camadas iniciais do BERT")

        elif loss_gap > 0.2 and acc_gap > 0.05:
            print("   ⚠️  OVERFITTING MODERADO DETECTADO")
            print("\n   ✅ SOLUÇÕES RECOMENDADAS:")
            print("   1. Aumentar dropout levemente (tente 0.2-0.3)")
            print("   2. Aumentar weight_decay (tente 0.02-0.05)")
            print("   3. Usar early stopping")

        elif final_train_loss > 0.8 and final_val_loss > 0.8:
            print("   ⚠️  UNDERFITTING DETECTADO!")
            print("   O modelo não está aprendendo adequadamente.")
            print("\n   ✅ SOLUÇÕES RECOMENDADAS:")
            print("   1. REDUZIR dropout (tente 0.05-0.1)")
            print("   2. REDUZIR weight_decay (tente 0.001-0.01)")
            print("   3. AUMENTAR learning rate (tente 3e-5 a 5e-5)")
            print("   4. Treinar por mais épocas (15-20)")
            print("   5. Verificar preprocessamento dos dados")
            print("   6. Aumentar batch size se possível")

        elif final_val_loss > 1.0:
            print("   ⚠️  VALIDATION LOSS MUITO ALTA!")
            print("   Possíveis causas:")
            print("   - Dados de validação muito diferentes dos dados de treino")
            print("   - Preprocessamento inadequado")
            print("   - Classes desbalanceadas")
            print("\n   ✅ SOLUÇÕES RECOMENDADAS:")
            print("   1. Verificar distribuição de classes")
            print("   2. Usar class_weights para balanceamento")
            print("   3. Melhorar preprocessamento (remover ruído, normalizar)")
            print("   4. Usar focal loss ou label smoothing")

        else:
            print("   ✅ MODELO RAZOAVELMENTE BALANCEADO")
            print("   Ainda há margem para melhorias:")
            print("   1. Ajuste fino de hiperparâmetros")
            print("   2. Experimente diferentes learning rates")
            print("   3. Use data augmentation para melhorar generalização")

        # Tendências
        print(f"\n📈 TENDÊNCIAS:")
        if len(val_losses) > 1:
            val_loss_trend = val_losses[-1] - val_losses[0]
            if val_loss_trend > 0:
                print(f"   ⚠️  Validation loss AUMENTOU de {val_losses[0]:.4f} para {val_losses[-1]:.4f}")
                print("   Isso indica que o modelo começou a overfitar durante o treino")
            else:
                print(f"   ✅ Validation loss DIMINUIU de {val_losses[0]:.4f} para {val_losses[-1]:.4f}")

    def analyze_data_distribution(self, data_paths):
        """Analisa distribuição dos dados"""
        print("\n" + "="*60)
        print("📊 ANÁLISE DE DISTRIBUIÇÃO DOS DADOS")
        print("="*60)

        for path in data_paths:
            if not os.path.exists(path):
                print(f"⚠️  Arquivo não encontrado: {path}")
                continue

            print(f"\n📁 Analisando: {path}")

            try:
                # Tenta carregar como CSV
                df = pd.read_csv(path)

                print(f"   Total de amostras: {len(df)}")

                if 'sentiment' in df.columns:
                    sentiment_counts = df['sentiment'].value_counts()
                    print(f"\n   Distribuição de sentimentos:")
                    for sentiment, count in sentiment_counts.items():
                        percentage = (count / len(df)) * 100
                        print(f"      {sentiment}: {count} ({percentage:.2f}%)")

                    # Verifica desbalanceamento
                    max_count = sentiment_counts.max()
                    min_count = sentiment_counts.min()
                    imbalance_ratio = max_count / min_count

                    if imbalance_ratio > 2.0:
                        print(f"\n   ⚠️  DESBALANCEAMENTO DETECTADO! Ratio: {imbalance_ratio:.2f}")
                        print("   ✅ RECOMENDAÇÕES:")
                        print("      - Ativar class_weights no config")
                        print("      - Usar oversampling/undersampling")
                        print("      - Usar focal loss")
                    else:
                        print(f"\n   ✅ Classes razoavelmente balanceadas (ratio: {imbalance_ratio:.2f})")

                if 'text' in df.columns:
                    text_lengths = df['text'].str.len()
                    word_counts = df['text'].str.split().str.len()

                    print(f"\n   Estatísticas de texto:")
                    print(f"      Comprimento médio (chars): {text_lengths.mean():.1f}")
                    print(f"      Comprimento médio (words): {word_counts.mean():.1f}")
                    print(f"      Max palavras: {word_counts.max()}")
                    print(f"      Min palavras: {word_counts.min()}")

                    # Verifica textos muito curtos ou longos
                    very_short = (word_counts < 3).sum()
                    very_long = (word_counts > 100).sum()

                    if very_short > 0:
                        print(f"\n   ⚠️  {very_short} textos com menos de 3 palavras")
                    if very_long > 0:
                        print(f"   ⚠️  {very_long} textos com mais de 100 palavras")
                        print(f"      Considere aumentar max_length no config")

            except Exception as e:
                print(f"   ❌ Erro ao carregar dados: {e}")

    def generate_recommendations(self, metrics):
        """Gera recomendações de configuração"""
        print("\n" + "="*60)
        print("⚙️  RECOMENDAÇÕES DE CONFIGURAÇÃO")
        print("="*60)

        print("""
📝 Para OVERFITTING (validation loss muito maior que training loss):

training:
  batch_size: 16 ou 32
  learning_rate: 1e-5 a 2e-5  # Reduzir
  weight_decay: 0.05 a 0.1    # Aumentar
  dropout_prob: 0.3 a 0.5     # Aumentar
  num_epochs: 10 a 15
  warmup_proportion: 0.1
  max_grad_norm: 1.0
  use_class_weights: true     # Se classes desbalanceadas
  early_stopping:
    patience: 3
    min_delta: 0.001

---

📝 Para UNDERFITTING (ambas losses altas):

training:
  batch_size: 16 ou 32
  learning_rate: 3e-5 a 5e-5  # Aumentar
  weight_decay: 0.001 a 0.01  # Reduzir
  dropout_prob: 0.05 a 0.1    # Reduzir
  num_epochs: 15 a 20         # Aumentar
  warmup_proportion: 0.15
  max_grad_norm: 1.0

---

📝 Para VALIDATION LOSS ALTA (> 1.0):

1. Verificar preprocessamento dos dados
2. Ativar class_weights
3. Melhorar limpeza de dados
4. Usar data augmentation
5. Experimentar learning rate menor (1e-5)
        """)

def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de modelo e dados")
    parser.add_argument("--experiment", type=str, help="Diretório do experimento")
    parser.add_argument("--train-data", type=str, help="Caminho para dados de treino")
    parser.add_argument("--val-data", type=str, help="Caminho para dados de validação")

    args = parser.parse_args()

    diagnostics = ModelDiagnostics(args.experiment)

    # Diagnóstico de métricas
    if args.experiment:
        metrics = diagnostics.load_metrics()
        if metrics:
            diagnostics.diagnose_overfitting(metrics)
            diagnostics.generate_recommendations(metrics)

    # Análise de dados
    data_paths = []
    if args.train_data:
        data_paths.append(args.train_data)
    if args.val_data:
        data_paths.append(args.val_data)

    if data_paths:
        diagnostics.analyze_data_distribution(data_paths)

    if not args.experiment and not data_paths:
        print("❌ Por favor, forneça --experiment ou --train-data/--val-data")
        print("\nExemplos de uso:")
        print("  python diagnose_model.py --experiment experiments/bert_base")
        print("  python diagnose_model.py --train-data dataset/processed/bert_train.csv --val-data dataset/processed/bert_validation.csv")
        print("  python diagnose_model.py --experiment experiments/bert_base --train-data dataset/processed/bert_train.csv")

if __name__ == "__main__":
    main()
