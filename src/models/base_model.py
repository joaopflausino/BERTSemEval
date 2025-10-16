import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseTransformerModel(nn.Module, ABC):
    def __init__(self, model_name: str, num_labels: int = 3, dropout_prob: float = 0.1):
        super(BaseTransformerModel, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob
        
        self.transformer = self._load_transformer()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = self._build_classifier()
        
    @abstractmethod
    def _load_transformer(self):
        """Load the specific transformer model"""
        pass
        
    @abstractmethod 
    def _build_classifier(self):
        """Build the classification head"""
        pass
        
    @abstractmethod
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """Forward pass through the model"""
        pass
        
    def get_model_info(self):
        """Return model information for comparison"""
        return {
            'model_name': self.model_name,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'hidden_size': self.transformer.config.hidden_size,
            'num_layers': self.transformer.config.num_hidden_layers,
            'num_attention_heads': self.transformer.config.num_attention_heads
        }