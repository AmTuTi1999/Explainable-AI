import hydra
import os
import sys
from omegaconf import DictConfig    
from src.xai_pipeline import explain_pipeline
@hydra.main(version_base= None, config_path="configs", config_name="explain.yaml")
def main(cfg: DictConfig):
    """Run the heart disease training pipeline.

    Args:
        cfg (DictConfig): Configuration for the experiment including model, data, and cross-validation parameters.
    """
    # Initialize and run the training pipeline
    explain_pipeline(cfg)
    # Run the main function

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Set this to get full error messages
    print(sys.path)
    main()
    # Run the main function
    # main()