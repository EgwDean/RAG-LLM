import os
import yaml

CONFIG_PATH = "config.yaml"

def load_config():
    """Load configuration from config.yaml."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at: {os.path.abspath(CONFIG_PATH)}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
