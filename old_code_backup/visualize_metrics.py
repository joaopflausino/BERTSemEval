import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    Plota e salva a matriz de confusão
    """
    plt.figure(figsize=(8, 6))
    
    # Normalizar a matriz de confusão para porcentagens
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Criar anotações com valores absolutos e porcentagens
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)'
    
    # Plotar heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Número de Amostras'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")

def plot_per_class_metrics(metrics_dict, class_names, title, save_path):
    """
    Plota métricas por classe (precision, recall, f1-score)
    """
    metrics = ['precision', 'recall', 'f1_score']
    n_classes = len(class_names)
    
    # Preparar dados
    data = {metric: [] for metric in metrics}
    for class_name in class_names:
        for metric in metrics:
            data[metric].append(metrics_dict[class_name][metric])
    
    # Configurar o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_classes)
    width = 0.25
    
    # Criar barras
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[metric], width, 
                      label=metric.replace('_', '-').capitalize(), color=color)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Configurar o gráfico
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Métricas por classe salvas em: {save_path}")

def plot_training_curves(all_metrics, num_epochs, save_path):
    """
    Plota curvas de treinamento e validação
    """
    epochs = list(range(num_epochs))
    
    # Extrair métricas
    train_loss = []
    val_loss = []
    train_f1 = []
    val_f1 = []
    train_acc = []
    val_acc = []
    
    for epoch in range(num_epochs):
        epoch_data = all_metrics[f'epoch_{epoch}']
        train_loss.append(epoch_data['training_loss'])
        val_loss.append(epoch_data['validation_loss'])
        train_f1.append(epoch_data['training_f1_measure'])
        val_f1.append(epoch_data['validation_f1_measure'])
        train_acc.append(epoch_data['training_accuracy'])
        val_acc.append(epoch_data['validation_accuracy'])
    
    # Criar subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_loss, 'b-', label='Treino', linewidth=2, marker='o')
    ax1.plot(epochs, val_loss, 'r-', label='Validação', linewidth=2, marker='s')
    ax1.set_title('Evolução da Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1-Score
    ax2.plot(epochs, train_f1, 'b-', label='Treino', linewidth=2, marker='o')
    ax2.plot(epochs, val_f1, 'r-', label='Validação', linewidth=2, marker='s')
    ax2.set_title('Evolução do F1-Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('F1-Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy
    ax3.plot(epochs, train_acc, 'b-', label='Treino', linewidth=2, marker='o')
    ax3.plot(epochs, val_acc, 'r-', label='Validação', linewidth=2, marker='s')
    ax3.set_title('Evolução da Acurácia', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Acurácia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparação de métricas finais
    best_epoch = all_metrics['best_epoch']
    final_metrics = {
        'Acurácia': all_metrics[f'epoch_{best_epoch}']['validation_accuracy'],
        'F1-Score': all_metrics[f'epoch_{best_epoch}']['validation_f1_measure'],
        'Recall': all_metrics[f'epoch_{best_epoch}']['validation_avg_recall'],
        'Precision': all_metrics[f'epoch_{best_epoch}']['validation_avg_precision']
    }
    
    bars = ax4.bar(final_metrics.keys(), final_metrics.values(), 
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax4.set_title(f'Métricas Finais de Validação (Época {best_epoch})', 
                  fontsize=12, fontweight='bold')
    ax4.set_ylabel('Valor')
    ax4.set_ylim(0, 1.0)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Análise de Desempenho do Modelo BERT para Análise de Sentimentos', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curvas de treinamento salvas em: {save_path}")

def generate_summary_report(all_metrics, save_path):
    """
    Gera um relatório resumido em formato de imagem
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    best_epoch = all_metrics['best_epoch']
    best_metrics = all_metrics[f'epoch_{best_epoch}']
    
    # Preparar dados do relatório
    report_data = [
        ['Métrica', 'Treino', 'Validação'],
        ['', '', ''],
        ['Acurácia', f"{best_metrics['training_accuracy']:.4f}", 
         f"{best_metrics['validation_accuracy']:.4f}"],
        ['F1-Score (weighted)', f"{best_metrics['training_f1_measure']:.4f}", 
         f"{best_metrics['validation_f1_measure']:.4f}"],
        ['Recall (macro)', f"{best_metrics['training_avg_recall']:.4f}", 
         f"{best_metrics['validation_avg_recall']:.4f}"],
        ['Precision (macro)', f"{best_metrics['training_avg_precision']:.4f}", 
         f"{best_metrics['validation_avg_precision']:.4f}"],
        ['Loss', f"{best_metrics['training_loss']:.4f}", 
         f"{best_metrics['validation_loss']:.4f}"],
        ['', '', ''],
        ['Informações Gerais', '', ''],
        ['Melhor Época', f'{best_epoch + 1}', ''],
        ['Duração do Treinamento', all_metrics['training_duration'], ''],
        ['Memória Máxima (MB)', f"{all_metrics['peak_cpu_memory_MB']:.0f}", '']
    ]
    
    # Criar tabela
    table = ax.table(cellText=report_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Estilizar cabeçalho
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Estilizar seções
    table[(8, 0)].set_facecolor('#2196F3')
    table[(8, 0)].set_text_props(weight='bold', color='white')
    
    plt.title('Relatório de Desempenho - Análise de Sentimentos BERT\nDataset SemEval 2013-2017', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Relatório resumido salvo em: {save_path}")

def plot_dataset_distribution_from_cm(confusion_matrices, class_names, dataset_name, save_path):
    """
    Plota distribuição de classes baseada na matriz de confusão
    """
    # Somar linhas da matriz de confusão para obter distribuição real
    cm = np.array(confusion_matrices)
    class_counts = cm.sum(axis=1)
    total = class_counts.sum()
    percentages = (class_counts / total) * 100
    
    # Cores para cada classe
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    plt.figure(figsize=(10, 7))
    
    # Criar barras
    bars = plt.bar(class_names, class_counts, color=colors, 
                    edgecolor='black', linewidth=1.5)
    
    # Adicionar valores e porcentagens
    for bar, count, pct in zip(bars, class_counts, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(count):,}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title(f'Distribuição de Classes - {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Classe de Sentimento', fontsize=14)
    plt.ylabel('Número de Amostras', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Linha de referência para distribuição balanceada
    balanced_count = total / 3
    plt.axhline(y=balanced_count, color='gray', linestyle='--', alpha=0.5, 
                label=f'Distribuição balanceada ({balanced_count:.0f})')
    plt.legend()
    
    plt.ylim(0, max(class_counts) * 1.15)
    
    # Estatísticas
    stats_text = f'Total: {int(total):,} amostras\n'
    max_idx = np.argmax(class_counts)
    min_idx = np.argmin(class_counts)
    stats_text += f'Majoritária: {class_names[max_idx]} ({percentages[max_idx]:.1f}%)\n'
    stats_text += f'Minoritária: {class_names[min_idx]} ({percentages[min_idx]:.1f}%)'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualizar métricas do modelo")
    parser.add_argument("--metrics_file", type=str, default="output/metrics.json",
                        help="Caminho para o arquivo metrics.json")
    parser.add_argument("--output_dir", type=str, default="output/visualizations",
                        help="Diretório para salvar as visualizações")
    args = parser.parse_args()
    
    # Criar diretório de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar métricas
    with open(args.metrics_file, 'r') as f:
        all_metrics = json.load(f)
    
    print("Gerando visualizações...")
    
    # Obter informações básicas
    num_epochs = all_metrics['training_epochs']
    best_epoch = all_metrics['best_epoch']
    class_names = all_metrics['class_names']
    
    # 1. Matriz de confusão da melhor época
    best_epoch_data = all_metrics[f'epoch_{best_epoch}']
    cm = np.array(best_epoch_data['validation_confusion_matrix'])
    plot_confusion_matrix(
        cm, class_names, 
        f'Matriz de Confusão - Teste (Época {best_epoch + 1})',
        output_dir / 'confusion_matrix_validation.png'
    )
    
    # 2. Métricas por classe
    plot_per_class_metrics(
        best_epoch_data['validation_per_class_metrics'],
        class_names,
        f'Métricas por Classe - Validação (Época {best_epoch + 1})',
        output_dir / 'metrics_per_class_validation.png'
    )
    
    # 3. Curvas de treinamento
    plot_training_curves(
        all_metrics, num_epochs,
        output_dir / 'training_curves.png'
    )
    
    # 4. Relatório resumido
    generate_summary_report(
        all_metrics,
        output_dir / 'summary_report.png'
    )
    
    # 5. Matriz de confusão do treino (última época)
    last_epoch = num_epochs - 1
    last_epoch_data = all_metrics[f'epoch_{last_epoch}']
    cm_train = np.array(last_epoch_data['training_confusion_matrix'])
    plot_confusion_matrix(
        cm_train, class_names,
        f'Matriz de Confusão - Treino (Época {last_epoch + 1})',
        output_dir / 'confusion_matrix_training.png'
    )
    
    # 6. Distribuição de classes no dataset de validação
    plot_dataset_distribution_from_cm(
        best_epoch_data['validation_confusion_matrix'],
        class_names,
        'Dataset de Teste (SemEval 2017)',
        output_dir / 'validation_class_distribution.png'
    )
    
    # 7. Distribuição de classes no dataset de treino
    plot_dataset_distribution_from_cm(
        last_epoch_data['training_confusion_matrix'],
        class_names,
        'Dataset de Treino (SemEval 2013-2016)',
        output_dir / 'training_class_distribution.png'
    )
    
    print(f"\nTodas as visualizações foram salvas em: {output_dir}")
    print("\nArquivos gerados:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()