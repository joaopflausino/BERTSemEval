from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from .dataset_reader import SemEvalDataset
from ..utils.class_weighting import compute_class_weights, print_class_analysis

def create_data_loaders(config):
    """Create training and evaluation data loaders"""
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    train_dataset = SemEvalDataset(
        config['data']['train_path'],
        tokenizer,
        max_length=config['data']['max_length']
    )

    eval_dataset = SemEvalDataset(
        config['data']['eval_path'],
        tokenizer,
        max_length=config['data']['max_length']
    )

    # Extract training labels for class weighting
    train_labels = train_dataset.labels

    # Analyze class distribution and compute weights if enabled
    class_weights = None
    if config.get('training', {}).get('use_class_weights', False):
        print_class_analysis(train_labels)
        method = config.get('training', {}).get('class_weight_method', 'balanced')
        class_weights = compute_class_weights(train_labels, method=method)
        print(f"\nUsing class weighting method: {method}")
        print("Class weights will be applied during training.")
    else:
        print("\nClass weighting is disabled.")

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config['training']['batch_size']
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=config['training']['batch_size']
    )

    return train_dataloader, eval_dataloader, tokenizer, train_labels, class_weights