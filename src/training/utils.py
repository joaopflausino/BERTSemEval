import random
import numpy as np
import torch
import os

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the appropriate device (CUDA/ROCm or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        
        # Check if ROCm is being used
        if "AMD" in torch.cuda.get_device_name() or os.environ.get('ROC_VISIBLE_DEVICES'):
            print("Detected AMD GPU with ROCm")
            # ROCm specific optimizations
            torch.backends.cudnn.benchmark = False  # May help with AMD GPUs
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    return device

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0