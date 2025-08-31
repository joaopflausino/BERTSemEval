import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

class SemEvalDataset(Dataset):
    
    def __init__(self, file_paths, tokenizer, max_length=128):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(file_paths)
        
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
            data = []
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('ID'): 
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 3:
                    
                        tweet_id, sentiment, text = parts[0], parts[1], parts[2]
                    
                        if sentiment == "positive":
                            label = 2
                        elif sentiment == "neutral":
                            label = 1
                        elif sentiment == "negative":
                            label = 0
                        else:
                        
                            continue
                        
                        data.append({
                            'id': tweet_id,
                            'text': text,
                            'label': label
                        })
            
            all_data.extend(data)
        
        return all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': item['label']
        }