import torch
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, device, class_names=['negative', 'neutral', 'positive']):
        self.model = model
        self.device = device
        self.class_names = class_names
        
    def evaluate(self, dataloader, save_path=None):
        """Comprehensive model evaluation"""
        self.model.eval()
        predictions = []
        labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_detailed_metrics(labels, predictions, probabilities)
        
        # Save results if path provided
        if save_path:
            self._save_evaluation_results(metrics, predictions, labels, probabilities, save_path)
        
        return metrics
    
    def _calculate_detailed_metrics(self, labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision_macro = precision_score(labels, predictions, average='macro')
        recall_macro = recall_score(labels, predictions, average='macro')
        
        # Per-class metrics
        f1_per_class = f1_score(labels, predictions, average=None)
        precision_per_class = precision_score(labels, predictions, average=None)
        recall_per_class = recall_score(labels, predictions, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(labels, predictions, target_names=self.class_names, output_dict=True)
        
        return {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i])
                }
                for i in range(len(self.class_names))
            },
            'classification_report': report
        }
    
    def _save_evaluation_results(self, metrics, predictions, labels, probabilities, save_path):
        """Save detailed evaluation results"""
        # Save metrics
        with open(f"{save_path}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions with probabilities
        results_df = pd.DataFrame({
            'true_label': labels,
            'predicted_label': predictions,
            'correct': [l == p for l, p in zip(labels, predictions)],
            'prob_negative': [prob[0] for prob in probabilities],
            'prob_neutral': [prob[1] for prob in probabilities],
            'prob_positive': [prob[2] for prob in probabilities]
        })
        
        results_df.to_csv(f"{save_path}_predictions.csv", index=False)
        
    def predict(self, dataloader):
        """Get predictions for a dataset"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        return all_predictions, all_probabilities