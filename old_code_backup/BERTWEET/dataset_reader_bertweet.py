import os
import re
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SemEvalDataset(Dataset):
    
    def __init__(self, file_paths, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(file_paths)
        
    def _preprocess_tweet(self, text):
        """
        Preprocess tweet text for BERTweet
        BERTweet has specific preprocessing requirements
        """
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # BERTweet specific preprocessing can be added here
        # The model is trained on raw tweets, so minimal preprocessing is needed
        
        return text
        
    def _load_data(self, file_paths):
        """Load data from files"""
        if isinstance(file_paths, str):
            if os.path.isdir(file_paths):
                files = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.txt')]
            else:
                files = [file_paths]
        else:
            files = file_paths
            
        all_data = []
        
        for file in files:
            print(f"Loading data from: {file}")
            data = []
            with open(file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('ID'): 
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        tweet_id, sentiment, text = parts[0], parts[1], parts[2]
                        
                        # Map sentiment labels to integers
                        if sentiment == "positive":
                            label = 2
                        elif sentiment == "neutral":
                            label = 1
                        elif sentiment == "negative":
                            label = 0
                        else:
                            print(f"Warning: Unknown sentiment '{sentiment}' in line {line_num}, skipping...")
                            continue
                        
                        # Preprocess tweet text
                        processed_text = self._preprocess_tweet(text)
                        
                        data.append({
                            'id': tweet_id,
                            'text': processed_text,
                            'original_text': text,
                            'label': label,
                            'sentiment': sentiment
                        })
                    else:
                        print(f"Warning: Invalid format in line {line_num}, expected at least 3 parts, got {len(parts)}")
            
            print(f"Loaded {len(data)} samples from {file}")
            all_data.extend(data)
        
        print(f"Total samples loaded: {len(all_data)}")
        
        # Print label distribution
        label_counts = {}
        for item in all_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("Label distribution:")
        print(f"  Negative (0): {label_counts.get(0, 0)}")
        print(f"  Neutral (1): {label_counts.get(1, 0)}")
        print(f"  Positive (2): {label_counts.get(2, 0)}")
        
        return all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize with BERTweet tokenizer
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': item['label'],
            'text': item['text'],
            'id': item['id']
        }
    
    def get_label_weights(self):
        """
        Calculate class weights for handling imbalanced datasets
        Returns tensor of weights for each class
        """
        import torch
        from collections import Counter
        
        labels = [item['label'] for item in self.data]
        label_counts = Counter(labels)
        
        total_samples = len(labels)
        num_classes = len(label_counts)
        
        # Calculate weights inversely proportional to class frequency
        weights = []
        for i in range(num_classes):
            if i in label_counts:
                weight = total_samples / (num_classes * label_counts[i])
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_info(self, idx):
        """Get detailed information about a specific sample"""
        item = self.data[idx]
        return {
            'index': idx,
            'id': item['id'],
            'text': item['text'],
            'original_text': item['original_text'],
            'label': item['label'],
            'sentiment': item['sentiment']
        }