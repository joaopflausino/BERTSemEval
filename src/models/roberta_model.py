import torch
import torch.nn as nn
from transformers import AutoModel
from .base_model import BaseTransformerModel

class RoBERTaSentimentClassifier(BaseTransformerModel):
    def __init__(self, model_name: str = "roberta-base", num_labels: int = 3, dropout_prob: float = 0.1, class_weights=None):
        super(RoBERTaSentimentClassifier, self).__init__(model_name, num_labels, dropout_prob, class_weights)
        
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
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        if labels is not None:
            if self.class_weights is not None:
                device = logits.device
                weights = self.class_weights.to(device)
                loss_fct = nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
            
        return logits