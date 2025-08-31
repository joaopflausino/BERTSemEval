#!/usr/bin/env python3
"""
Main training script for sentiment analysis models
Support for multiple transformer architectures with ROCm compatibility
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config_loader import load_config
from src.utils.model_factory import create_model
from src.data.data_loader import create_data_loaders
from src.training.trainer import ModelTrainer
from src.training.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis models")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Override output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if provided
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    
    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    print(f"Starting experiment: {config['experiment']['name']}")
    print(f"Model: {config['model']['name']}")
    print(f"Description: {config['experiment']['description']}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model(config)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Model info: {model.get_model_info()}")
    
    # Create data loaders
    print("\nPreparing data...")
    train_dataloader, eval_dataloader, tokenizer = create_data_loaders(config)
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Evaluation samples: {len(eval_dataloader.dataset)}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = ModelTrainer(
        model=model,
        config=config,
        output_dir=config['output']['output_dir']
    )
    
    # Train model
    print("\nStarting training...")
    metrics = trainer.train(train_dataloader, eval_dataloader)
    
    print(f"\nExperiment completed!")
    print(f"Results saved to: {config['output']['output_dir']}")
    
    return metrics

if __name__ == "__main__":
    main()