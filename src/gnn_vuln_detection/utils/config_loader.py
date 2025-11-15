import os
from pathlib import Path
from typing import Any

from .file_loader import load_yaml

CONFIG_DIR = Path(__file__).resolve(strict=True).parent.parent.parent.parent / "config"


def load_config(filename: str) -> dict[str, Any]:
    """Load a YAML configuration file from the config directory.

    Args:
        filename (str): Name of the YAML file to load (e.g. 'model_params.yaml').

    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path):
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    return load_yaml(path)


def load_all_configs() -> dict[str, dict[str, Any]]:
    """Load all YAML configuration files in the config directory.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of filename (without extension) to config dict.
    """
    configs = {}
    for fname in os.listdir(CONFIG_DIR):
        if fname.endswith((".yaml", ".yml")):
            key = os.path.splitext(fname)[0]
            configs[key] = load_config(fname)
    return configs


def get_cwe_labels() -> dict[int, str]:
    return load_config("model_params.yaml")["vulnerabilities"]
