import logging
import os
import sys

import hydra
from omegaconf import DictConfig
from src.pipelines import run_heartdisease_training


@hydra.main(version_base=None, config_path="configs", config_name="train_tabular")
def main(cfg: DictConfig):
    """Run the heart disease training pipeline.

    Args:
        cfg (DictConfig): Configuration for the experiment including model, data, and cross-validation parameters.
    """
    # Initialize and run the training pipeline
    run_heartdisease_training(cfg)

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Set this to get full error messages
    main()
    # Run the main function
    # main()