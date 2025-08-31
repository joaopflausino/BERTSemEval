import torch
import torch.nn as nn
from transformers import AutoModel
from .base_model import BaseTransformerModel

class DistilBertSentimentClassifier(BaseTransformerModel):
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 3, dropout_prob: float = 0.1):
        super(DistilBertSentimentClassifier, self).__init__(model_name, num_labels, dropout_prob)
    
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
        # DistilBERT doesn't support token_type_ids, so we ignore it
        # Only pass supported arguments to the transformer
        transformer_inputs = {
            'input_ids': input_ids,
        }
        
        # Only add attention_mask if it's provided
        if attention_mask is not None:
            transformer_inputs['attention_mask'] = attention_mask
            
        # Call transformer with only supported arguments
        outputs = self.transformer(**transformer_inputs)
        
        # DistilBERT doesn't have pooler_output, use CLS token
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Take CLS token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits