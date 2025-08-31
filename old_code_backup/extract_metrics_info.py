#!/usr/bin/env python
"""
Script para extrair informações específicas do arquivo metrics.json
Útil para incluir valores exatos no artigo
"""
import json
import argparse
from pathlib import Path

def print_section(title):
    """Imprime um cabeçalho de seção formatado"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print('='*50)

def extract_metrics_info(metrics_file):
    """Extrai e exibe informações importantes das métricas"""
    
    # Carregar métricas
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    best_epoch = metrics['best_epoch']
    best_data = metrics[f'epoch_{best_epoch}']
    
    # 1. Informações Gerais
    print_section("INFORMAÇÕES GERAIS")
    print(f"Melhor época: {best_epoch + 1}")
    print(f"Duração do treinamento: {metrics['training_duration']}")
    print(f"Memória máxima utilizada: {metrics['peak_cpu_memory_MB']:.0f} MB")
    
    # 2. Métricas da Melhor Época
    print_section("MÉTRICAS DA MELHOR ÉPOCA (VALIDAÇÃO)")
    print(f"Acurácia: {best_data['validation_accuracy']:.4f} ({best_data['validation_accuracy']*100:.2f}%)")
    print(f"F1-Score (weighted): {best_data['validation_f1_measure']:.4f}")
    print(f"Recall (macro): {best_data['validation_avg_recall']:.4f}")
    print(f"Precision (macro): {best_data['validation_avg_precision']:.4f}")
    print(f"Loss: {best_data['validation_loss']:.4f}")
    
    # 3. Métricas por Classe
    print_section("MÉTRICAS POR CLASSE (VALIDAÇÃO)")
    class_names = metrics['class_names']
    per_class = best_data['validation_per_class_metrics']
    
    print(f"{'Classe':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 40)
    for class_name in class_names:
        p = per_class[class_name]['precision']
        r = per_class[class_name]['recall']
        f1 = per_class[class_name]['f1_score']
        print(f"{class_name:<10} {p:<10.4f} {r:<10.4f} {f1:<10.4f}")
    
    # 4. Distribuição de Classes
    print_section("DISTRIBUIÇÃO DE CLASSES")
    
    # Calcular distribuição do dataset de validação
    val_cm = best_data['validation_confusion_matrix']
    val_totals = [sum(row) for row in val_cm]
    val_total = sum(val_totals)
    
    print("Dataset de Validação:")
    for i, class_name in enumerate(class_names):
        count = val_totals[i]
        pct = (count / val_total) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    # Calcular distribuição do dataset de treino (última época)
    last_epoch = metrics['training_epochs'] - 1
    train_cm = metrics[f'epoch_{last_epoch}']['training_confusion_matrix']
    train_totals = [sum(row) for row in train_cm]
    train_total = sum(train_totals)
    
    print("\nDataset de Treino:")
    for i, class_name in enumerate(class_names):
        count = train_totals[i]
        pct = (count / train_total) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    # 5. Principais Erros de Classificação
    print_section("PRINCIPAIS ERROS DE CLASSIFICAÇÃO")
    cm = best_data['validation_confusion_matrix']
    errors = []
    
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i][j] > 0:
                error_count = cm[i][j]
                error_pct = (error_count / val_totals[i]) * 100
                errors.append({
                    'true': true_class,
                    'pred': pred_class,
                    'count': error_count,
                    'pct': error_pct
                })
    
    # Ordenar por contagem de erros
    errors.sort(key=lambda x: x['count'], reverse=True)
    
    print("Top 5 confusões mais frequentes:")
    for i, error in enumerate(errors[:5]):
        print(f"{i+1}. {error['true']} → {error['pred']}: "
              f"{error['count']} erros ({error['pct']:.1f}% dos {error['true']})")
    
    # 6. Comparação entre Épocas
    print_section("EVOLUÇÃO POR ÉPOCA")
    print(f"{'Época':<8} {'Train F1':<10} {'Val F1':<10} {'Train Loss':<12} {'Val Loss':<10}")
    print("-" * 50)
    
    for epoch in range(metrics['training_epochs']):
        epoch_data = metrics[f'epoch_{epoch}']
        train_f1 = epoch_data['training_f1_measure']
        val_f1 = epoch_data['validation_f1_measure']
        train_loss = epoch_data['training_loss']
        val_loss = epoch_data['validation_loss']
        
        marker = " *" if epoch == best_epoch else ""
        print(f"{epoch+1:<8} {train_f1:<10.4f} {val_f1:<10.4f} "
              f"{train_loss:<12.4f} {val_loss:<10.4f}{marker}")
    
    print("\n* = Melhor época")
    
    # 7. Formato LaTeX para Tabelas
    print_section("FORMATO LATEX PARA INCLUSÃO NO ARTIGO")
    
    print("\n% Tabela de métricas por classe")
    print("\\begin{tabular}{l|ccc}")
    print("\\toprule")
    print("\\textbf{Classe} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\")
    print("\\midrule")
    for class_name in class_names:
        p = per_class[class_name]['precision']
        r = per_class[class_name]['recall']
        f1 = per_class[class_name]['f1_score']
        print(f"{class_name.capitalize()} & {p:.3f} & {r:.3f} & {f1:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
    print("\n% Valores para incluir no texto")
    print(f"% Acurácia: {best_data['validation_accuracy']*100:.1f}\\%")
    print(f"% F1-Score: {best_data['validation_f1_measure']:.3f}")
    print(f"% Melhor época: {best_epoch + 1}")

def main():
    parser = argparse.ArgumentParser(
        description="Extrai informações detalhadas do arquivo metrics.json"
    )
    parser.add_argument(
        "--metrics_file", 
        type=str, 
        default="output/metrics.json",
        help="Caminho para o arquivo metrics.json"
    )
    
    args = parser.parse_args()
    
    # Verificar se o arquivo existe
    if not Path(args.metrics_file).exists():
        print(f"Erro: Arquivo '{args.metrics_file}' não encontrado!")
        print("Execute primeiro o treinamento para gerar o arquivo metrics.json")
        return
    
    extract_metrics_info(args.metrics_file)

if __name__ == "__main__":
    main()