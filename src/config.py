# import config anywhere in the project:

# from src.config import load_config
# cfg = load_config()
# cfg["carla"]["host"]

import yaml
from pathlib import Path

def load_config(exp: str) -> dict:
    path = f"configs/{exp}.yaml"
    with open(Path(path)) as f:
        return yaml.safe_load(f)

