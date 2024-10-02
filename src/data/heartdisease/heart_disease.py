import pandas as pd
import torch
from omegaconf import DictConfig
from src.data.heartdisease.feature_preprocessor import FeaturePreprocessor  # Assuming the class is defined as above

def load_data(data_config: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Load, preprocess, scale, and split the data based on configuration."""
    # Step 1: Load the data from the URL
    url = data_config.url
    column_names = data_config.column_names
    categorical_columns = data_config.categorical_columns
    numerical_columns = data_config.numerical_columns

    # Load dataset
    data = pd.read_csv(url, names=column_names)

    # Step 2: Initialize the FeaturePreprocessor
    preprocessor = FeaturePreprocessor()

    # Step 3: Preprocess the data
    # Preprocess the data (apply numerical scaling and one-hot encoding)
    data_encoded = preprocessor.preprocess(data, categorical_columns, numerical_columns)

    # Step 4: Split the features (X) and target (y)
    X = data_encoded.drop('target', axis=1)  # Drop the target column from features
    y = (data_encoded['target'] > 0).astype(int)  # Convert target to binary (assuming a binary classification problem)

    # Step 5: Save the preprocessor (which includes the scaler and categorical mappings)
    preprocessor.save_preprocessor("preprocessor.pkl")

    # Return the processed features and target
    return torch.Tensor(X), torch.Tensor(y)
