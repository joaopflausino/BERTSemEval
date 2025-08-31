#!/usr/bin/env python3
"""
Script to run regularized experiments to fix overfitting
"""

import subprocess
import sys
from pathlib import Path

def run_experiment(config_path):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_path}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, "train_model.py", "--config", str(config_path)]
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✓ Successfully completed: {config_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {config_path}")
        print(f"Error: {e}")
        return False

def main():
    """Run all regularized experiments"""
    configs = [
        "configs/bert_base_regularized.yaml",
        "configs/roberta_base_regularized.yaml", 
        "configs/electra_base_regularized.yaml",
        "configs/distilbert_base_regularized.yaml",
        "configs/bertweet_base_regularized.yaml"
    ]
    
    print("Starting regularized experiments to fix overfitting...")
    print(f"Total experiments: {len(configs)}")
    
    successful = 0
    failed = 0
    
    for config in configs:
        if Path(config).exists():
            if run_experiment(config):
                successful += 1
            else:
                failed += 1
        else:
            print(f"✗ Config file not found: {config}")
            failed += 1
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {successful + failed}")
    
    if failed == 0:
        print("\n🎉 All experiments completed successfully!")
    else:
        print(f"\n⚠️  {failed} experiments failed")
        
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)