import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class ModelComparison:
    def __init__(self, results_dir="experiments"):
        self.results_dir = Path(results_dir)
        self.class_names = ['negative', 'neutral', 'positive']
        
    def load_experiment_results(self, experiment_names):
        """Load results from multiple experiments"""
        results = {}
        
        for exp_name in experiment_names:
            exp_path = self.results_dir / exp_name
            metrics_path = exp_path / "metrics.json"
            
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    results[exp_name] = json.load(f)
        
        return results
    
    def create_comparison_table(self, results):
        """Create a comparison table of model performance"""
        comparison_data = []
        
        for model_name, metrics in results.items():
            # Extract model info
            model_info = metrics.get('model_info', {})
            
            # Get best validation metrics
            best_epoch = metrics.get('best_epoch', 0)
            best_metrics = metrics.get(f'epoch_{best_epoch}', {})
            
            comparison_data.append({
                'Model': model_name,
                'Model Size': model_info.get('model_name', 'Unknown'),
                'Parameters (M)': f"{model_info.get('num_parameters', 0) / 1e6:.1f}",
                'Validation Accuracy': f"{best_metrics.get('validation_accuracy', 0):.4f}",
                'Validation F1 (Weighted)': f"{best_metrics.get('validation_f1_measure', 0):.4f}",
                'Validation F1 (Macro)': f"{best_metrics.get('validation_avg_recall', 0):.4f}",
                'Training Duration': metrics.get('training_duration', 'Unknown'),
                'Peak Memory (MB)': f"{metrics.get('peak_cpu_memory_MB', 0):.1f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_training_curves(self, results, save_path=None):
        """Plot training curves for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Curves Comparison', fontsize=16)
        
        metrics_to_plot = [
            ('training_loss', 'Training Loss'),
            ('validation_loss', 'Validation Loss'), 
            ('training_f1_measure', 'Training F1'),
            ('validation_f1_measure', 'Validation F1')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            for model_name, model_metrics in results.items():
                epochs = []
                values = []
                
                for epoch_key in model_metrics.keys():
                    if epoch_key.startswith('epoch_'):
                        epoch_num = int(epoch_key.split('_')[1])
                        if metric in model_metrics[epoch_key]:
                            epochs.append(epoch_num)
                            values.append(model_metrics[epoch_key][metric])
                
                if epochs:
                    ax.plot(epochs, values, label=model_name, marker='o')
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, results, save_path=None):
        """Create performance comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Extract data for comparison
        models = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for model_name, metrics in results.items():
            best_epoch = metrics.get('best_epoch', 0)
            best_metrics = metrics.get(f'epoch_{best_epoch}', {})
            
            models.append(model_name)
            accuracies.append(best_metrics.get('validation_accuracy', 0))
            f1_scores.append(best_metrics.get('validation_f1_measure', 0))
            precisions.append(best_metrics.get('validation_avg_precision', 0))
            recalls.append(best_metrics.get('validation_avg_recall', 0))
        
        # Plot comparisons
        metrics_data = [
            (accuracies, 'Validation Accuracy'),
            (f1_scores, 'Validation F1 Score'),
            (precisions, 'Validation Precision'),
            (recalls, 'Validation Recall')
        ]
        
        for idx, (data, title) in enumerate(metrics_data):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(models, data, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(title)
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Confusion Matrices Comparison', fontsize=16)
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(results.items()):
            ax = axes[idx] if n_models > 1 else axes[0]
            
            # Get confusion matrix from best epoch
            best_epoch = metrics.get('best_epoch', 0)
            best_metrics = metrics.get(f'epoch_{best_epoch}', {})
            cm = np.array(best_metrics.get('validation_confusion_matrix', [[0]]))
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=ax)
            ax.set_title(f'{model_name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_report(self, results, save_path="model_comparison_report.html"):
        """Generate a comprehensive HTML report"""
        # Create comparison table
        comparison_df = self.create_comparison_table(results)
        
        html_content = f"""
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 20px 0; }}
                .best {{ background-color: #90EE90; }}
            </style>
        </head>
        <body>
            <h1>Sentiment Analysis Model Comparison Report</h1>
            
            <h2>Model Performance Summary</h2>
            {comparison_df.to_html(index=False, classes='table', escape=False)}
            
            <h2>Detailed Analysis</h2>
        """
        
        # Find best performing model for each metric
        best_accuracy = max(results.items(), key=lambda x: x[1].get(f'epoch_{x[1].get("best_epoch", 0)}', {}).get('validation_accuracy', 0))
        best_f1 = max(results.items(), key=lambda x: x[1].get(f'epoch_{x[1].get("best_epoch", 0)}', {}).get('validation_f1_measure', 0))
        
        html_content += f"""
            <div class="metric">
                <h3>Best Accuracy: {best_accuracy[0]} ({best_accuracy[1].get(f'epoch_{best_accuracy[1].get("best_epoch", 0)}', {}).get('validation_accuracy', 0):.4f})</h3>
                <h3>Best F1 Score: {best_f1[0]} ({best_f1[1].get(f'epoch_{best_f1[1].get("best_epoch", 0)}', {}).get('validation_f1_measure', 0):.4f})</h3>
            </div>
            
            </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        return comparison_df