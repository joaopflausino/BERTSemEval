import json
import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from JSON or YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def save_config(config, config_path):
    """Save configuration to JSON or YAML file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def create_default_config(model_name="bert-base-uncased"):
    """Create a default configuration"""
    return {
        "experiment": {
            "name": f"{model_name.replace('/', '_')}_experiment",
            "description": "Sentiment analysis experiment",
            "seed": 42
        },
        "model": {
            "type": "bert" if "bert" in model_name.lower() else "auto",
            "name": model_name,
            "num_labels": 3,
            "dropout_prob": 0.1
        },
        "data": {
            "train_path": "dataset/train",
            "eval_path": "dataset/test/SemEval2017-task4-test.subtask-A.english.txt",
            "max_length": 128
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "num_epochs": 3,
            "warmup_proportion": 0.1,
            "max_grad_norm": 1.0
        },
        "output": {
            "output_dir": f"experiments/{model_name.replace('/', '_')}",
            "save_best_only": True
        }
    }