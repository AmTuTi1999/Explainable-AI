"""Heart_Disease_Data"""
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from src.data.heartdisease.feature_preprocessor import FeaturePreprocessor 

def load_data(data_config: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Load, preprocess, scale, and split the data based on configuration."""
    # Step 1: Load the data from the URL
    url = data_config.url
    column_names = data_config.columns
    categorical_columns = data_config.categorical_columns
    numerical_columns = data_config.numerical_columns
    

    data = pd.read_csv(url, names=column_names)
    preprocessor = FeaturePreprocessor()
    data_encoded = preprocessor.preprocess(data, categorical_columns, numerical_columns)
    X = data_encoded.drop('target', axis=1)  
    y = (data_encoded['target'] > data_encoded['target'].mean()).astype(int)  
    preprocessor.save_preprocessor("preprocessor.pkl")
    return torch.Tensor(X.to_numpy().astype(float)), torch.Tensor(y.to_numpy().astype(float))