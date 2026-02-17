"""Unified training script using generalized Trainer."""

import argparse
import yaml
import torch

from training import Trainer
from models import get_model
from data import get_data_loader
from utils import print_columns, logging_columns_list


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_value(value_str):
    """Parse a string value to appropriate Python type."""
    # Handle None/null
    if value_str.lower() in ('none', 'null', ''):
        return None
    
    # Handle booleans
    if value_str.lower() in ('true', 'yes', 'on'):
        return True
    if value_str.lower() in ('false', 'no', 'off'):
        return False
    
    # Handle numbers
    try:
        # Try int first
        if '.' not in value_str and 'e' not in value_str.lower():
            return int(value_str)
        # Then float
        return float(value_str)
    except ValueError:
        pass
    
    # Try parsing as YAML (for arrays, lists, etc.)
    try:
        parsed = yaml.safe_load(value_str)
        # Only return parsed value if it's not a string (to avoid double-wrapping)
        if not isinstance(parsed, str):
            return parsed
    except (yaml.YAMLError, ValueError):
        pass
    
    # Return as string if nothing else matches
    return value_str


def set_nested_value(config, key_path, value):
    """Set a nested value in config using dot-notation key path.
    
    Supports:
    - Simple keys: 'training.epochs'
    - List indices: 'training.optimizer_groups.0.optimizer.lr'
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for i, key in enumerate(keys[:-1]):
        # Check if key is a list index
        if key.isdigit():
            idx = int(key)
            if not isinstance(current, list) or idx >= len(current):
                raise ValueError(f"Cannot access list index {idx} in path '{key_path}'")
            current = current[idx]
        else:
            if key not in current:
                current[key] = {}
            current = current[key]
    
    # Set the final value
    final_key = keys[-1]
    if final_key.isdigit():
        idx = int(final_key)
        if not isinstance(current, list):
            raise ValueError(f"Cannot use index on non-list in path '{key_path}'")
        if idx >= len(current):
            # Extend list if needed
            current.extend([None] * (idx - len(current) + 1))
        current[idx] = value
    else:
        current[final_key] = value


def apply_overrides(config, overrides):
    """Apply command-line overrides to config.
    
    Args:
        config: The loaded YAML config dictionary
        overrides: List of strings in format 'key.path=value'
    """
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Override must be in format 'key=value', got: {override}")
        
        key_path, value_str = override.split('=', 1)
        key_path = key_path.strip()
        value_str = value_str.strip()
        
        value = parse_value(value_str)
        set_nested_value(config, key_path, value)
        
        print(f"Override: {key_path} = {value} (type: {type(value).__name__})")


def print_model_parameters(model):
    """Print total number of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def main():
    parser = argparse.ArgumentParser(description='Unified training with generalized Trainer')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of training runs to execute')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom name for the wandb run')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--override', '--param', dest='overrides', action='append',
                        default=[], metavar='KEY=VALUE',
                        help='Override config parameters. Can be used multiple times. '
                             'Supports nested keys with dot notation, e.g., '
                             '"training.epochs=200" or "training.optimizer_groups.0.optimizer.lr=0.001"')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Apply command-line overrides
    if args.overrides:
        print(f"\nApplying {len(args.overrides)} override(s):")
        apply_overrides(config, args.overrides)
        print()
    
    model_config = config['model']
    model = get_model(model_config['name'])()
    model = model.to(args.device)
    
    # Print model parameters
    print_model_parameters(model)
    
    if args.device == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        # model.compile(mode="max-autotune")
    
    dataset_config = config['dataset']
    train_loader = get_data_loader(dataset_config['name'])(
        dataset_config['path'],
        train=True,
        **dataset_config.get('train_args', {})
    )
    test_loader = get_data_loader(dataset_config['name'])(
        dataset_config['path'],
        train=False,
        **dataset_config.get('test_args', {})
    )
    
    if hasattr(model, 'reset'):
        model.reset()
    
    if config['training'].get('warmup', True):
        warmup_trainer = Trainer(model, config, device=args.device)
        print_columns(logging_columns_list, is_head=True)
        warmup_trainer.train(
            train_loader,
            test_loader,
            wandb_enabled=False,
            wandb_config=None,
            warmup=True
        )    
    accuracies = []
    
    for run in range(args.runs):
        if hasattr(model, 'reset'):
            model.reset()
        
        trainer = Trainer(model, config, device=args.device)
        trainer.run_number = run  # Set run number for table display
        
        wandb_enabled = not args.no_wandb
        wandb_config = None
        if wandb_enabled:
            wandb_config = {
                "project": config['training']['wandb']['project'],
                "config": {
                    **config,
                    'run_number': run
                },
                "job_type": "train"
            }
            if args.run_name:
                wandb_config["name"] = f"{args.run_name}_run{run}"
        
        final_acc = trainer.train(
            train_loader,
            test_loader,
            wandb_enabled=wandb_enabled,
            wandb_config=wandb_config
        )
        
        accuracies.append(final_acc)
    
    if args.runs > 1:
        accuracies_tensor = torch.tensor(accuracies)
        print(f"\n{'='*60}")
        print(f"Summary over {args.runs} runs:")
        print(f"Mean accuracy: {accuracies_tensor.mean():.4f}")
        print(f"Std accuracy: {accuracies_tensor.std():.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

