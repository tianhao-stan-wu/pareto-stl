# import config anywhere in the project:

import yaml
from pathlib import Path

def load_config(exp: str) -> dict:
    path = f"configs/{exp}.yaml"
    with open(Path(path)) as f:
        return yaml.safe_load(f)

