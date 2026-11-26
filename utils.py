"""Load the experiment configuration without importing the training stack."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "bci2a.yaml"


def load_config(path=None):
    """Read YAML. Relative paths inside the configuration are resolved by callers."""
    config_path = DEFAULT_CONFIG if path is None else Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    for section in ("data", "model", "paths", "training", "validation", "normalization"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping.")
    return config
