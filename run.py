#!/usr/bin/env python3
"""
Central orchestration script for BERTSemEval framework
Provides organized procedures to run different models and experiments
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

class BERTSemEvalRunner:
    """Main runner class for organizing all procedures"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.available_configs = [
            "configs/bert_base.yaml",
            "configs/roberta_base.yaml", 
            "configs/distilbert_base.yaml",
            "configs/electra_base.yaml"
        ]
        self.scripts = {
            "train": "train_model.py",
            "experiments": "run_experiments.py", 
            "compare": "compare_models.py"
        }

    def list_available_models(self):
        """List all available model configurations"""
        print("Available Models:")
        print("-" * 50)
        for i, config in enumerate(self.available_configs, 1):
            config_path = self.project_root / config
            if config_path.exists():
                model_name = config_path.stem
                print(f"{i:2d}. {model_name.upper():<15} ({config})")
            else:
                print(f"{i:2d}. {config_path.stem.upper():<15} (MISSING: {config})")

    def train_single_model(self, config: str, output_dir: Optional[str] = None):
        """Train a single model with given configuration"""
        config_path = self.project_root / config
        
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config}")
            return False
            
        print(f"Training model with config: {config}")
        print("-" * 50)
        
        cmd = [sys.executable, self.scripts["train"], "--config", str(config_path)]
        if output_dir:
            cmd.extend(["--output_dir", output_dir])
            
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✅ Training completed successfully for {config}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for {config} (exit code: {e.returncode})")
            return False

    def run_all_experiments(self):
        """Run all available model experiments in sequence"""
        print("Running all experiments in batch mode...")
        print("-" * 50)
        
        try:
            result = subprocess.run([sys.executable, self.scripts["experiments"]], check=True)
            print("✅ All experiments completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Batch experiments failed (exit code: {e.returncode})")
            return False

    def compare_models(self, experiments: List[str], generate_report: bool = False):
        """Compare results from multiple experiments"""
        if len(experiments) < 2:
            print("Error: Need at least 2 experiments for comparison")
            return False
            
        print(f"Comparing models: {', '.join(experiments)}")
        print("-" * 50)
        
        cmd = [sys.executable, self.scripts["compare"], "--experiments"] + experiments
        if generate_report:
            cmd.append("--report")
            
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Model comparison completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Model comparison failed (exit code: {e.returncode})")
            return False

    def interactive_mode(self):
        """Interactive mode for user-guided execution"""
        print("\n" + "="*60)
        print("BERTSemEval Framework - Interactive Mode")
        print("="*60)
        
        while True:
            print("\nChoose an option:")
            print("1. List available models")
            print("2. Train a single model")
            print("3. Run all experiments")
            print("4. Compare trained models") 
            print("5. Train specific models and compare")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                self.list_available_models()
                
            elif choice == "2":
                self.list_available_models()
                model_choice = input("\nEnter model number or config path: ").strip()
                
                if model_choice.isdigit():
                    model_idx = int(model_choice) - 1
                    if 0 <= model_idx < len(self.available_configs):
                        config = self.available_configs[model_idx]
                    else:
                        print("Invalid model number")
                        continue
                else:
                    config = model_choice
                    
                self.train_single_model(config)
                
            elif choice == "3":
                self.run_all_experiments()
                
            elif choice == "4":
                experiments = input("Enter experiment names (space-separated): ").strip().split()
                if experiments:
                    generate_report = input("Generate HTML report? (y/n): ").lower() == 'y'
                    self.compare_models(experiments, generate_report)
                    
            elif choice == "5":
                self.list_available_models()
                model_choices = input("Enter model numbers (space-separated): ").strip().split()
                
                configs = []
                for choice in model_choices:
                    if choice.isdigit():
                        model_idx = int(choice) - 1
                        if 0 <= model_idx < len(self.available_configs):
                            configs.append(self.available_configs[model_idx])
                
                if len(configs) < 2:
                    print("Need at least 2 models for comparison")
                    continue
                    
                print(f"\nTraining {len(configs)} models...")
                successful_experiments = []
                
                for config in configs:
                    if self.train_single_model(config):
                        exp_name = Path(config).stem
                        successful_experiments.append(exp_name)
                
                if len(successful_experiments) >= 2:
                    generate_report = input("Generate HTML report? (y/n): ").lower() == 'y'
                    self.compare_models(successful_experiments, generate_report)
                    
            elif choice == "6":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(
        description="BERTSemEval Framework Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run.py
  
  # List available models
  python run.py --list
  
  # Train a single model
  python run.py --train configs/bert_base.yaml
  
  # Run all experiments
  python run.py --all
  
  # Compare specific models
  python run.py --compare bert_base roberta_base --report
  
  # Train and compare specific models
  python run.py --train-compare configs/bert_base.yaml configs/roberta_base.yaml
        """
    )
    
    parser.add_argument("--list", action="store_true",
                       help="List available model configurations")
    parser.add_argument("--train", type=str,
                       help="Train a single model with given config file")
    parser.add_argument("--output-dir", type=str,
                       help="Override output directory for training")
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments in sequence")
    parser.add_argument("--compare", nargs='+',
                       help="Compare results from specified experiments")
    parser.add_argument("--report", action="store_true",
                       help="Generate HTML report for comparisons")
    parser.add_argument("--train-compare", nargs='+',
                       help="Train specified models and compare them")
    
    args = parser.parse_args()
    
    runner = BERTSemEvalRunner()
    
    # If no arguments provided, start interactive mode
    if not any(vars(args).values()):
        runner.interactive_mode()
        return
    
    # List available models
    if args.list:
        runner.list_available_models()
        return
    
    # Train single model
    if args.train:
        success = runner.train_single_model(args.train, args.output_dir)
        sys.exit(0 if success else 1)
    
    # Run all experiments
    if args.all:
        success = runner.run_all_experiments()
        sys.exit(0 if success else 1)
    
    # Compare models
    if args.compare:
        success = runner.compare_models(args.compare, args.report)
        sys.exit(0 if success else 1)
    
    # Train and compare models
    if args.train_compare:
        if len(args.train_compare) < 2:
            print("Error: Need at least 2 models for train-compare")
            sys.exit(1)
            
        print(f"Training and comparing {len(args.train_compare)} models...")
        successful_experiments = []
        
        for config in args.train_compare:
            if runner.train_single_model(config):
                exp_name = Path(config).stem
                successful_experiments.append(exp_name)
        
        if len(successful_experiments) >= 2:
            success = runner.compare_models(successful_experiments, args.report)
            sys.exit(0 if success else 1)
        else:
            print("Not enough successful experiments for comparison")
            sys.exit(1)

if __name__ == "__main__":
    main()