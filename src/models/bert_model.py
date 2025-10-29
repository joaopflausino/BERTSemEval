import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .base_model import BaseTransformerModel

class BertSentimentClassifier(BaseTransformerModel):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3, dropout_prob: float = 0.1, class_weights=None, loss_fn=None):
        self.loss_fn = loss_fn
        super(BertSentimentClassifier, self).__init__(model_name, num_labels, dropout_prob, class_weights)
        
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
            # Use custom loss function if provided
            if self.loss_fn is not None:
                loss = self.loss_fn(logits, labels)
            elif self.class_weights is not None:
                device = logits.device
                weights = self.class_weights.to(device)
                loss_fct = nn.CrossEntropyLoss(weight=weights)
                loss = loss_fct(logits, labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            return loss, logits

        return logits

    def freeze_bert_layers(self, num_layers_to_freeze=None):
        """
        Freeze BERT layers to prevent overfitting

        Args:
            num_layers_to_freeze: Number of layers to freeze from the bottom.
                                 If None, freezes embeddings only.
        """
        if num_layers_to_freeze is None:
            # Freeze only embeddings
            for param in self.transformer.embeddings.parameters():
                param.requires_grad = False
        else:
            # Freeze embeddings + specified number of layers
            for param in self.transformer.embeddings.parameters():
                param.requires_grad = False

            for i in range(min(num_layers_to_freeze, len(self.transformer.encoder.layer))):
                for param in self.transformer.encoder.layer[i].parameters():
                    param.requires_grad = False

    def unfreeze_all_layers(self):
        """Unfreeze all BERT layers"""
        for param in self.transformer.parameters():
            param.requires_grad = True