import torch
import torch.nn as nn
from transformers import BertModel

class BertSentimentClassifier(nn.Module):
    
    def __init__(self, bert_model_name="bert-base-uncased", num_labels=3, dropout_prob=0.5):

        super(BertSentimentClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        
        
        # self.fc1 = nn.Linear(self.bert.config.hidden_size, 768)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(768, 384)
        # self.dropout2 = nn.Dropout(0.5)

        
        self.fc1 = nn.Linear(self.bert.config.hidden_size,self.bert.config.hidden_size // 2)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(768, 384)
        self.dropout2 = nn.Dropout(0.5)

        
        
        self.classifier = nn.Linear(self.bert.config.hidden_size // 2, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        x = self.dropout2(x)
        
        logits = self.classifier(x)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits