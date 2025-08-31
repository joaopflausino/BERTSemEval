import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_metrics(metrics_history, save_path=None):
    """Plot training metrics over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    epochs = list(range(len(metrics_history['train_loss'])))
    
    # Training and validation loss
    axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Training')
    axes[0, 0].plot(epochs, metrics_history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training and validation accuracy
    axes[0, 1].plot(epochs, metrics_history['train_acc'], 'b-', label='Training')
    axes[0, 1].plot(epochs, metrics_history['val_acc'], 'r-', label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training and validation F1
    axes[1, 0].plot(epochs, metrics_history['train_f1'], 'b-', label='Training')
    axes[1, 0].plot(epochs, metrics_history['val_f1'], 'r-', label='Validation')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate (if available)
    if 'learning_rate' in metrics_history:
        axes[1, 1].plot(epochs, metrics_history['learning_rate'], 'g-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_class_distribution(labels, class_names, save_path=None):
    """Plot distribution of classes in dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, alpha=0.7)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_model_comparison(results_dict, metric='f1_score', save_path=None):
    """Compare multiple models on a specific metric"""
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, scores, alpha=0.7)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.xlabel('Model')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()