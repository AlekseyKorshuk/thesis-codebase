import hashlib
import yaml


def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def hash_sample(sample):
    text = "\n".join([f"{key}: {value}" for key, value in sample.items()])
    return hashlib.sha256(text.encode()).hexdigest()
