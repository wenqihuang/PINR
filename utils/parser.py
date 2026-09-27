
import yaml
import os
from types import SimpleNamespace


def dict_to_namespace(d):
    """Convert dictionary to a nested SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def namespace_to_dict(namespace):
    """Convert a SimpleNamespace back to a dictionary."""
    if isinstance(namespace, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(namespace).items()}
    return namespace

def load_config(config_path: str):
    """Load configuration from a YAML file with support for nested structures and object-style access."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file) or {}
    
    config_struct = dict_to_namespace(config_dict)
    
    return config_dict, config_struct

