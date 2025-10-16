import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .base_model import BaseTransformerModel

class BertSentimentClassifier(BaseTransformerModel):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3, dropout_prob: float = 0.1):
        super(BertSentimentClassifier, self).__init__(model_name, num_labels, dropout_prob)
        
    def _load_transformer(self):
        return AutoModel.from_pretrained(self.model_name)
        
    def _build_classifier(self):
        hidden_size = self.transformer.config.hidden_size
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(hidden_size // 2, self.num_labels)
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        transformer_inputs = {'input_ids': input_ids}
        
        if attention_mask is not None:
            transformer_inputs['attention_mask'] = attention_mask
        
        if token_type_ids is not None and hasattr(self.transformer.config, 'type_vocab_size'):
            transformer_inputs['token_type_ids'] = token_type_ids
            
        outputs = self.transformer(**transformer_inputs)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
            
        return logits