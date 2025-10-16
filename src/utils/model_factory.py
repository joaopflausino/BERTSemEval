from ..models import (
    BertSentimentClassifier,
    RoBERTaSentimentClassifier, 
    DistilBertSentimentClassifier,
    ElectraSentimentClassifier,
    BertTweetSentimentClassifier
)

def create_model(config):
    """Factory function to create models based on config"""
    model_type = config['model']['type'].lower()
    model_name = config['model']['name']
    num_labels = config['model']['num_labels']
    dropout_prob = config['model']['dropout_prob']
    
    if model_type == 'bert' or 'bert' in model_name.lower():
        return BertSentimentClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout_prob=dropout_prob
        )
    elif model_type == 'roberta' or 'roberta' in model_name.lower():
        return RoBERTaSentimentClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout_prob=dropout_prob
        )
    elif model_type == 'distilbert' or 'distilbert' in model_name.lower():
        return DistilBertSentimentClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout_prob=dropout_prob
        )
    elif model_type == 'electra' or 'electra' in model_name.lower():
        return ElectraSentimentClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout_prob=dropout_prob
        )
    elif model_type == 'bertweet' or 'bertweet' in model_name.lower():
        return BertTweetSentimentClassifier(
            bertweet_model_name=model_name,
            num_labels=num_labels,
            dropout_prob=dropout_prob
        )
    else:
        # Default to BERT for unknown types
        return BertSentimentClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout_prob=dropout_prob
        )

def get_supported_models():
    """Return list of supported model architectures"""
    return {
        'bert': ['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased'],
        'roberta': ['roberta-base', 'roberta-large'],
        'distilbert': ['distilbert-base-uncased', 'distilbert-base-cased'],
        'electra': ['google/electra-base-discriminator', 'google/electra-small-discriminator'],
        'bertweet': ['vinai/bertweet-base']
    }