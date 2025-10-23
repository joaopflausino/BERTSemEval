import torch
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

def analyze_class_distribution(labels):
    """
    Analyze class distribution in the dataset

    Args:
        labels (list): List of class labels (0, 1, 2)

    Returns:
        dict: Class distribution information
    """
    label_counts = Counter(labels)
    total_samples = len(labels)

    class_names = ['negative', 'neutral', 'positive']

    distribution = {
        'total_samples': total_samples,
        'class_counts': dict(label_counts),
        'class_percentages': {
            label: (count / total_samples) * 100
            for label, count in label_counts.items()
        },
        'class_names': {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
    }

    return distribution

def compute_class_weights(labels, method='balanced'):
    """
    Compute class weights for imbalanced datasets

    Args:
        labels (list): List of class labels
        method (str): Method to compute weights ('balanced', 'inverse_freq')

    Returns:
        torch.Tensor: Class weights tensor
    """
    unique_labels = sorted(list(set(labels)))

    if method == 'balanced':
        # Use sklearn's balanced class weight computation
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(unique_labels),
            y=np.array(labels)
        )
    elif method == 'inverse_freq':
        # Simple inverse frequency weighting
        label_counts = Counter(labels)
        total_samples = len(labels)
        class_weights = [
            total_samples / (len(unique_labels) * label_counts[label])
            for label in unique_labels
        ]
    else:
        raise ValueError(f"Unknown method: {method}")

    return torch.tensor(class_weights, dtype=torch.float32)

def print_class_analysis(labels):
    """
    Print detailed class distribution analysis

    Args:
        labels (list): List of class labels
    """
    distribution = analyze_class_distribution(labels)

    print("\n" + "="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    print(f"Total samples: {distribution['total_samples']}")
    print("\nClass distribution:")

    for label, count in distribution['class_counts'].items():
        class_name = distribution['class_names'][label]
        percentage = distribution['class_percentages'][label]
        print(f"  {class_name} ({label}): {count:,} samples ({percentage:.1f}%)")

    # Calculate imbalance ratio
    max_count = max(distribution['class_counts'].values())
    min_count = min(distribution['class_counts'].values())
    imbalance_ratio = max_count / min_count

    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")

    # Compute and display class weights
    weights_balanced = compute_class_weights(labels, method='balanced')
    weights_inverse = compute_class_weights(labels, method='inverse_freq')

    print("\nClass weights (balanced method):")
    for i, weight in enumerate(weights_balanced):
        class_name = distribution['class_names'][i]
        print(f"  {class_name} ({i}): {weight:.4f}")

    print("\nClass weights (inverse frequency method):")
    for i, weight in enumerate(weights_inverse):
        class_name = distribution['class_names'][i]
        print(f"  {class_name} ({i}): {weight:.4f}")

    print("="*50)

    return distribution, weights_balanced, weights_inverse