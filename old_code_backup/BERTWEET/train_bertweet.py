import os
import time
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
import psutil
import argparse

from dataset_reader_bertweet import SemEvalDataset
from model_bertweet import BertTweetSentimentClassifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)
    
    # Use AutoTokenizer for BERTweet
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    train_dataset = SemEvalDataset(
        args.train_dir,
        tokenizer,
        max_length=args.max_seq_length
    )
    eval_dataset = SemEvalDataset(
        args.eval_file,
        tokenizer,
        max_length=args.max_seq_length
    )
    print(f"Loaded {len(train_dataset)} training examples")
    print(f"Loaded {len(eval_dataset)} evaluation examples")
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size
    )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.batch_size
    )
    
    model = BertTweetSentimentClassifier(
        bertweet_model_name=args.model_name,
        num_labels=3,
        dropout_prob=args.dropout
    )
    model.to(device)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0.0
    best_epoch = 0
    process = psutil.Process(os.getpid())
    peak_memory = 0
    all_metrics = {
        "training_start_epoch": 0,
        "training_epochs": args.num_epochs,
        "model_name": args.model_name
    }
    
    start_time = time.time()
    patience_counter = 0
    for epoch in range(args.num_epochs):
        print(f"\n======== Epoch {epoch+1} / {args.num_epochs} ========")
        print("\nTraining...")
        model.train()
        total_train_loss = 0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_dataloader): 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss.backward()          
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)    
            optimizer.step()
            scheduler.step()         
            
            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, memory)
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        train_recall = recall_score(train_labels, train_preds, average='macro')
        
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Training accuracy: {train_accuracy:.4f}")
        print(f"  Training F1: {train_f1:.4f}")
        print(f"  Training recall: {train_recall:.4f}")   
        
        print("\nEvaluating...")
        model.eval()
        total_eval_loss = 0
        eval_preds, eval_labels = [], []
        
        for batch in tqdm(eval_dataloader):   
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) 
            
            with torch.no_grad():
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            total_eval_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            eval_preds.extend(preds)
            eval_labels.extend(labels.cpu().numpy())
        
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        eval_accuracy = accuracy_score(eval_labels, eval_preds)
        eval_f1 = f1_score(eval_labels, eval_preds, average='weighted')
        eval_recall = recall_score(eval_labels, eval_preds, average='macro')
        
        print(f"  Average evaluation loss: {avg_eval_loss:.4f}")
        print(f"  Evaluation accuracy: {eval_accuracy:.4f}")
        print(f"  Evaluation F1: {eval_f1:.4f}")
        print(f"  Evaluation recall: {eval_recall:.4f}")
        #ANTIGO
        #if eval_f1 > best_val_f1:
        #    best_val_f1 = eval_f1
        #    best_epoch = epoch.
        if eval_f1 > best_val_f1:
            best_val_f1 = eval_f1
            best_epoch = epoch
            patience_counter = 0
            
            print(f"  New best model! Saving to {args.output_dir}")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            
            # Save tokenizer and model config
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
                json.dump({
                    "model_name": args.model_name,
                    "num_labels": 3,
                    "dropout_prob": args.dropout,
                    "max_seq_length": args.max_seq_length
                }, f, indent=2)
        else:
            patience_counter += 1  # Incrementa quando não melhora
            print(f"  No improvement. Patience: {patience_counter}/2")

        if patience_counter >= 2:  # Para após 2 épocas sem melhoria
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        epoch_metrics = {
            f"epoch_{epoch}": {
                "training_accuracy": float(train_accuracy),
                "training_avg_recall": float(train_recall),
                "training_f1_measure": float(train_f1),
                "training_loss": float(avg_train_loss),
                "validation_accuracy": float(eval_accuracy),
                "validation_avg_recall": float(eval_recall),
                "validation_f1_measure": float(eval_f1),
                "validation_loss": float(avg_eval_loss)
            }
        }
        
        all_metrics.update(epoch_metrics)    
    
    training_duration = time.time() - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    all_metrics.update({
        "best_epoch": best_epoch,
        "peak_cpu_memory_MB": float(peak_memory),
        "training_duration": formatted_duration,
        "epoch": args.num_epochs - 1,
        "training_accuracy": float(train_accuracy),
        "training_avg_recall": float(train_recall),
        "training_f1_measure": float(train_f1),
        "training_loss": float(avg_train_loss),
        "training_cpu_memory_MB": float(process.memory_info().rss / (1024 * 1024)),
        "validation_accuracy": float(eval_accuracy),
        "validation_avg_recall": float(eval_recall),
        "validation_f1_measure": float(eval_f1),
        "validation_loss": float(avg_eval_loss),
        "best_validation_accuracy": float(all_metrics[f"epoch_{best_epoch}"]["validation_accuracy"]),
        "best_validation_avg_recall": float(all_metrics[f"epoch_{best_epoch}"]["validation_avg_recall"]),
        "best_validation_f1_measure": float(all_metrics[f"epoch_{best_epoch}"]["validation_f1_measure"]),
        "best_validation_loss": float(all_metrics[f"epoch_{best_epoch}"]["validation_loss"])
    })
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best model saved from epoch {best_epoch+1}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Training duration: {formatted_duration}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="vinai/bertweet-base", type=str,
                        help="Pre-trained BERTweet model name")
    parser.add_argument("--train_dir", default="dataset/train", type=str,
                        help="Path to training data directory")
    parser.add_argument("--eval_file", default="dataset/test/SemEval2017-task4-test.subtask-A.english.txt", type=str,
                        help="Path to evaluation data file")
    parser.add_argument("--output_dir", default="output_bertweet", type=str,
                        help="Path to output directory")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="Learning rate (lower for BERTweet)")
    parser.add_argument("--weight_decay", default=0.02, type=float,
                        help="Weight decay")
    parser.add_argument("--num_epochs", default=3, type=int,
                        help="Number of training epochs")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training steps for learning rate warmup")
    parser.add_argument("--max_grad_norm", default=0.5, type=float,
                        help="Maximum gradient norm")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout probability")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed")
    
    args = parser.parse_args()

    print("BERTweet Configuration:")
    print(args)
    train(args)