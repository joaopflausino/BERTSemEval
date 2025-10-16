import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from transformers import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from dataset_reader import SemEvalDataset
from model import BertSentimentClassifier

def analyze_errors(model, dataloader, device, class_names, save_dir):
    """
    Analisa erros do modelo e gera visualizações
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 1. Análise de confiança por classe
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        plt.subplot(2, 2, i+1)
        
        # Separar predições corretas e incorretas
        correct_mask = (all_preds == i) & (all_labels == i)
        incorrect_mask = (all_preds == i) & (all_labels != i)
        
        if np.sum(correct_mask) > 0:
            correct_probs = all_probs[correct_mask, i]
            plt.hist(correct_probs, bins=20, alpha=0.7, label='Corretas', color='green', density=True)
        
        if np.sum(incorrect_mask) > 0:
            incorrect_probs = all_probs[incorrect_mask, i]
            plt.hist(incorrect_probs, bins=20, alpha=0.7, label='Incorretas', color='red', density=True)
        
        plt.xlabel('Probabilidade')
        plt.ylabel('Densidade')
        plt.title(f'Distribuição de Confiança - {class_name.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Análise de Confiança das Predições por Classe', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Matriz de confusão detalhada com erros mais comuns
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    
    # Calcular erros mais comuns
    error_matrix = cm.copy()
    np.fill_diagonal(error_matrix, 0)
    
    # Top 3 erros mais comuns
    error_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and error_matrix[i, j] > 0:
                error_pairs.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': error_matrix[i, j],
                    'percentage': (error_matrix[i, j] / cm[i].sum()) * 100
                })
    
    error_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    # Plotar matriz de erros
    sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Número de Erros'})
    
    plt.title('Matriz de Erros de Classificação', fontsize=14, fontweight='bold')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribuição de classes no dataset
    plt.figure(figsize=(8, 6))
    
    unique, counts = np.unique(all_labels, return_counts=True)
    percentages = (counts / len(all_labels)) * 100
    
    bars = plt.bar(class_names, counts, color=['#d62728', '#ff7f0e', '#2ca02c'])
    
    # Adicionar valores e porcentagens nas barras
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
    
    plt.title('Distribuição de Classes no Dataset de Teste', fontsize=14, fontweight='bold')
    plt.xlabel('Classe')
    plt.ylabel('Número de Amostras')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Análise de erros concluída!")
    return all_preds, all_labels, all_probs

def generate_performance_heatmap(all_preds, all_labels, class_names, save_dir):
    """
    Gera um heatmap de performance por classe
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    # Calcular métricas
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Criar DataFrame
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }, index=class_names)
    
    # Plotar heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Valor'})
    
    plt.title('Heatmap de Performance por Classe', fontsize=14, fontweight='bold')
    plt.xlabel('Classe')
    plt.ylabel('Métrica')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analisar predições do modelo")
    parser.add_argument("--model_path", type=str, default="output/model.pt",
                        help="Caminho para o modelo salvo")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Nome do modelo BERT pré-treinado")
    parser.add_argument("--eval_file", type=str, 
                        default="dataset/test/SemEval2017-task4-test.subtask-A.english.txt",
                        help="Arquivo de validação")
    parser.add_argument("--output_dir", type=str, default="output/analysis",
                        help="Diretório para salvar análises")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Tamanho do batch")
    args = parser.parse_args()
    
    # Criar diretório
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Carregar tokenizer e dataset
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    eval_dataset = SemEvalDataset(args.eval_file, tokenizer, max_length=128)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    
    # Carregar modelo
    model = BertSentimentClassifier(
        bert_model_name=args.model_name,
        num_labels=3,
        dropout_prob=0.1
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    class_names = ['negative', 'neutral', 'positive']
    
    print("Analisando predições...")
    all_preds, all_labels, all_probs = analyze_errors(
        model, eval_dataloader, device, class_names, output_dir
    )
    
    print("Gerando heatmap de performance...")
    generate_performance_heatmap(all_preds, all_labels, class_names, output_dir)
    
    print(f"\nAnálises salvas em: {output_dir}")

if __name__ == "__main__":
    main()