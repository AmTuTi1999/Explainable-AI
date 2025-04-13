import pytest
import os
from omegaconf import OmegaConf
from src.pipelines import run_heartdisease_training  # Adjust if it's in a different path


def mock_cfg():
    # Path to your config file in the 'config/models/' folder
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'models', 'tabular_models', 'heart_disease.yaml')  # Adjust path if needed

    # Load the YAML file with OmegaConf
    cfg = OmegaConf.load(config_path)

    return cfg

def main():
    cfg = mock_cfg()
    run_heartdisease_training(cfg)

if __name__ == "__main__":
    main()