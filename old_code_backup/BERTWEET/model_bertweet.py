import torch
import torch.nn as nn
from transformers import AutoModel

class BertTweetSentimentClassifier(nn.Module):
    
    def __init__(self, bertweet_model_name="vinai/bertweet-base", num_labels=3, dropout_prob=0.3):

        super(BertTweetSentimentClassifier, self).__init__()
        
        # Use AutoModel for BERTweet
        self.bertweet = AutoModel.from_pretrained(bertweet_model_name)
        self.dropout = nn.Dropout(0.4)
        
        # Get the hidden size from the model config
        hidden_size = self.bertweet.config.hidden_size
        
        # Architecture with intermediate layer
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_size // 2, num_labels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights of the classifier layers"""
        for module in [self.fc1, self.classifier]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        
        # BERTweet forward pass
        outputs = self.bertweet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use pooler output (CLS token representation)
        pooled_output = outputs.pooler_output
        
        # Apply classification layers
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        logits = self.classifier(x)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits
    
    def freeze_bertweet_layers(self, num_layers_to_freeze=None):
        """
        Freeze BERTweet layers for fine-tuning
        Args:
            num_layers_to_freeze: Number of layers to freeze from the bottom.
                                 If None, freezes embeddings only.
        """
        if num_layers_to_freeze is None:
            # Freeze only embeddings
            for param in self.bertweet.embeddings.parameters():
                param.requires_grad = False
        else:
            # Freeze embeddings
            for param in self.bertweet.embeddings.parameters():
                param.requires_grad = False
            
            # Freeze specified number of encoder layers
            for i in range(min(num_layers_to_freeze, len(self.bertweet.encoder.layer))):
                for param in self.bertweet.encoder.layer[i].parameters():
                    param.requires_grad = False
    
    def unfreeze_all_layers(self):
        """Unfreeze all BERTweet layers"""
        for param in self.bertweet.parameters():
            param.requires_grad = True