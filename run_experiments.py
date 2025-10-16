#!/usr/bin/env python3
"""
Script to run multiple experiments in sequence for model comparison
"""

import subprocess
import sys
import time
from pathlib import Path

def run_experiment(config_path):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Starting experiment with config: {config_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "train_model.py", 
            "--config", str(config_path)
        ], check=True, capture_output=True, text=True)
        
        duration = time.time() - start_time
        print(f"Experiment completed successfully in {duration:.2f} seconds")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with return code {e.returncode}")
        print("STDOUT:", e.stdout[-500:])
        print("STDERR:", e.stderr[-500:])
        return False

def main():
    # List of configuration files to run
    configs = [
        "configs/bert_base.yaml",
        "configs/roberta_base.yaml", 
        "configs/distilbert_base.yaml",
        "configs/electra_base.yaml"
    ]
    
    print("Starting batch experiment run...")
    print(f"Will run {len(configs)} experiments:")
    for config in configs:
        print(f"  - {config}")
    
    # Verify all config files exist
    missing_configs = []
    for config in configs:
        if not Path(config).exists():
            missing_configs.append(config)
    
    if missing_configs:
        print(f"Missing configuration files: {missing_configs}")
        return
    
    # Run experiments
    results = {}
    total_start_time = time.time()
    
    for config in configs:
        success = run_experiment(config)
        results[config] = success
        
        if not success:
            print(f"Experiment {config} failed. Continue? (y/n): ", end="")
            response = input().lower()
            if response != 'y':
                print("Stopping experiment batch.")
                break
    
    # Summary
    total_duration = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("EXPERIMENT BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"Results:")
    
    successful_experiments = []
    for config, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {config}: {status}")
        if success:
            # Extract experiment name from config
            exp_name = Path(config).stem
            successful_experiments.append(exp_name)
    
    # Run comparison if we have successful experiments
    if len(successful_experiments) >= 2:
        print(f"\nRunning model comparison for successful experiments...")
        try:
            subprocess.run([
                sys.executable, "compare_models.py",
                "--experiments"] + successful_experiments + [
                "--report"
            ], check=True)
            print("Model comparison completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Model comparison failed: {e}")
    else:
        print("Need at least 2 successful experiments for comparison.")

if __name__ == "__main__":
    main()