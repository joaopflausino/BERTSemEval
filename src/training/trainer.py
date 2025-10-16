import os
import time
import json
import torch
import psutil
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from tqdm import tqdm
from .utils import get_device, get_memory_usage

class ModelTrainer:
    def __init__(self, model, config, output_dir):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device = get_device()
        self.class_names = ['negative', 'neutral', 'positive']
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
    def _setup_optimizer(self):
        """Setup optimizer with weight decay"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['training']['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=float(self.config['training']['learning_rate'])
        )
    
    def _setup_scheduler(self, total_steps):
        """Setup learning rate scheduler"""
        warmup_steps = int(total_steps * self.config['training']['warmup_proportion'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def train(self, train_dataloader, eval_dataloader):
        """Main training loop"""
        total_steps = len(train_dataloader) * self.config['training']['num_epochs']
        self._setup_scheduler(total_steps)
        
        best_val_f1 = 0.0
        best_epoch = 0
        process = psutil.Process(os.getpid())
        peak_memory = 0
        
        # Early stopping parameters
        early_stopping_config = self.config['training'].get('early_stopping', {})
        patience = early_stopping_config.get('patience', float('inf'))
        min_delta = early_stopping_config.get('min_delta', 0.0)
        patience_counter = 0
        
        all_metrics = {
            "training_start_epoch": 0,
            "training_epochs": self.config['training']['num_epochs'],
            "model_info": self.model.get_model_info()
        }
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\n======== Epoch {epoch+1} / {self.config['training']['num_epochs']} ========")
            
            # Training phase
            train_metrics = self._train_epoch(train_dataloader)
            
            # Evaluation phase  
            eval_metrics = self._evaluate_epoch(eval_dataloader)
            
            # Memory tracking
            memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, memory)
            gpu_memory = get_memory_usage()
            
            # Print metrics
            self._print_epoch_metrics(epoch, train_metrics, eval_metrics)
            
            # Save best model and check early stopping
            if eval_metrics['f1'] > best_val_f1 + min_delta:
                best_val_f1 = eval_metrics['f1']
                best_epoch = epoch
                patience_counter = 0
                self._save_model()
                print(f"  New best model saved! F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epochs")
                
                if patience_counter >= patience:
                    print(f"  Early stopping triggered! No improvement for {patience} epochs")
                    break
            
            # Store epoch metrics
            epoch_data = {
                f"epoch_{epoch}": {
                    **{f"training_{k}": v for k, v in train_metrics.items()},
                    **{f"validation_{k}": v for k, v in eval_metrics.items()},
                    "gpu_memory_gb": gpu_memory,
                    "cpu_memory_mb": memory
                }
            }
            all_metrics.update(epoch_data)
        
        # Final metrics
        training_duration = time.time() - start_time
        hours, remainder = divmod(training_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        final_metrics = {
            "best_epoch": best_epoch,
            "peak_cpu_memory_MB": float(peak_memory),
            "training_duration": formatted_duration,
            "best_validation_f1_measure": float(best_val_f1),
            "class_names": self.class_names
        }
        all_metrics.update(final_metrics)
        
        # Save metrics
        self._save_metrics(all_metrics)
        
        print(f"\nTraining complete!")
        print(f"Best model saved from epoch {best_epoch+1}")
        print(f"Best validation F1: {best_val_f1:.4f}")
        print(f"Training duration: {formatted_duration}")
        
        return all_metrics
    
    def _train_epoch(self, train_dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions, labels = [], []
        
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            batch_labels = batch['labels'].to(self.device)
            
            self.model.zero_grad()
            
            loss, logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['max_grad_norm']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            labels.extend(batch_labels.cpu().numpy())
        
        return self._calculate_metrics(labels, predictions, total_loss / len(train_dataloader))
    
    def _evaluate_epoch(self, eval_dataloader):
        """Evaluate for one epoch"""
        self.model.eval()
        total_loss = 0
        predictions, labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)
                
                loss, logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                labels.extend(batch_labels.cpu().numpy())
        
        return self._calculate_metrics(labels, predictions, total_loss / len(eval_dataloader))
    
    def _calculate_metrics(self, labels, predictions, loss):
        """Calculate evaluation metrics"""
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='macro')
        precision = precision_score(labels, predictions, average='macro')
        
        # Per-class metrics
        f1_per_class = f1_score(labels, predictions, average=None)
        precision_per_class = precision_score(labels, predictions, average=None)
        recall_per_class = recall_score(labels, predictions, average=None)
        cm = confusion_matrix(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'loss': loss,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i])
                }
                for i in range(len(self.class_names))
            }
        }
    
    def _print_epoch_metrics(self, epoch, train_metrics, eval_metrics):
        """Print epoch metrics"""
        print(f"  Training - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Validation - Loss: {eval_metrics['loss']:.4f}, "
              f"Acc: {eval_metrics['accuracy']:.4f}, F1: {eval_metrics['f1']:.4f}")
        
        print("\n  Per-class validation metrics:")
        for class_name, metrics in eval_metrics['per_class_metrics'].items():
            print(f"    {class_name}: P={metrics['precision']:.4f}, "
                  f"R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
    
    def _save_model(self):
        """Save the best model"""
        os.makedirs(self.output_dir, exist_ok=True)
        model_path = os.path.join(self.output_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
    
    def _save_metrics(self, metrics):
        """Save training metrics"""
        os.makedirs(self.output_dir, exist_ok=True)
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")