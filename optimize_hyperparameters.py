#!/usr/bin/env python3
"""
Hyperparameter optimization using Optuna
"""

import optuna
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config_loader import load_config, save_config
from src.utils.model_factory import create_model
from src.data.data_loader import create_data_loaders
from src.training.trainer import ModelTrainer
from src.training.utils import set_seed


def objective(trial, base_config_path, param_ranges):
    """Optuna objective function"""
    # Load base config
    config = load_config(base_config_path)

    # Suggest hyperparameters
    for param_name, param_config in param_ranges.items():
        keys = param_name.split('.')

        # Get suggested value
        if param_config['type'] == 'categorical':
            value = trial.suggest_categorical(param_name, param_config['choices'])
        elif param_config['type'] == 'float':
            value = trial.suggest_float(param_name, param_config['low'], param_config['high'],
                                      log=param_config.get('log', False))
        elif param_config['type'] == 'int':
            value = trial.suggest_int(param_name, param_config['low'], param_config['high'])

        # Set value in config
        current_config = config
        for key in keys[:-1]:
            if key not in current_config:
                current_config[key] = {}
            current_config = current_config[key]
        current_config[keys[-1]] = value

    # Update output directory for this trial
    config['output']['output_dir'] = f"optuna_trial_{trial.number}"

    try:
        # Set seed
        set_seed(config['experiment']['seed'])

        # Create model and data loaders
        model = create_model(config)
        train_dataloader, eval_dataloader, tokenizer, train_labels, class_weights = create_data_loaders(config)

        # Train
        trainer = ModelTrainer(model, config, config['output']['output_dir'])
        metrics = trainer.train(train_dataloader, eval_dataloader)

        # Return F1 score
        return metrics.get('best_validation_f1_measure', 0.0)

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def optimize(base_config_path, n_trials=50, param_ranges=None):
    """Run hyperparameter optimization"""

    if param_ranges is None:
        param_ranges = {
            'training.learning_rate': {'type': 'float', 'low': 1e-5, 'high': 5e-5, 'log': True},
            'training.batch_size': {'type': 'categorical', 'choices': [8, 16, 32]},
            'training.weight_decay': {'type': 'float', 'low': 0.0, 'high': 0.1},
            'training.warmup_proportion': {'type': 'float', 'low': 0.05, 'high': 0.2},
            'model.dropout_prob': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'training.num_epochs': {'type': 'int', 'low': 3, 'high': 6}
        }

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, base_config_path, param_ranges), n_trials=n_trials)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save best config
    best_config = load_config(base_config_path)
    for param_name, value in study.best_params.items():
        keys = param_name.split('.')
        current_config = best_config
        for key in keys[:-1]:
            current_config = current_config[key]
        current_config[keys[-1]] = value

    best_config['output']['output_dir'] = "best_optimized_model"
    save_config(best_config, "best_optimized_config.json")
    print("Best config saved to: best_optimized_config.json")

    return study


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Base config path')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    args = parser.parse_args()

    optimize(args.config, args.trials)