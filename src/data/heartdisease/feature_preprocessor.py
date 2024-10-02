import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

class FeaturePreprocessor:
    def __init__(self):
        self.numerical_scaler = None
        self.categorical_columns = None
        self.numerical_columns = None

    def preprocess(self, data: pd.DataFrame, categorical_columns: list, numerical_columns: list):
        """
        Main preprocess function that handles both numerical and categorical transformations.
        Args:
            data (pd.DataFrame): The input dataset to preprocess.
            categorical_columns (list): List of categorical columns.
            numerical_columns (list): List of numerical columns.
        Returns:
            pd.DataFrame: The preprocessed data.
        """
        # Store categorical and numerical columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        # Numerical transformation (scaling)
        data[self.numerical_columns] = self._transform_numerical(data[self.numerical_columns])

        # Categorical transformation (one-hot encoding)
        data = self._transform_categorical(data, categorical_columns)

        return data

    def _transform_numerical(self, numerical_data: pd.DataFrame):
        """
        Apply scaling transformation to numerical columns.
        Args:
            numerical_data (pd.DataFrame): Data containing numerical features.
        Returns:
            pd.DataFrame: Scaled numerical features.
        """
        self.numerical_scaler = StandardScaler()
        return pd.DataFrame(self.numerical_scaler.fit_transform(numerical_data), columns=numerical_data.columns)

    def _transform_categorical(self, data: pd.DataFrame, categorical_columns: list):
        """
        Apply one-hot encoding to categorical columns.
        Args:
            data (pd.DataFrame): The original dataset.
            categorical_columns (list): List of categorical columns.
        Returns:
            pd.DataFrame: Data with one-hot encoded categorical features.
        """
        return pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    def inverse_transform_numerical(self, scaled_data: pd.DataFrame):
        """
        Inverse the scaling transformation on numerical columns.
        Args:
            scaled_data (pd.DataFrame): Scaled numerical features.
        Returns:
            pd.DataFrame: Original numerical features.
        """
        return pd.DataFrame(self.numerical_scaler.inverse_transform(scaled_data), columns=scaled_data.columns)

    def save_preprocessor(self, path: str = "preprocessor.pkl"):
        """
        Save the numerical scaler and column information.
        Args:
            path (str): Path to save the preprocessor object.
        """
        with open(path, "wb") as f:
            pickle.dump({
                'numerical_scaler': self.numerical_scaler,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns
            }, f)

    def load_preprocessor(self, path: str = "preprocessor.pkl"):
        """
        Load the preprocessor object (scaler and column info) from a pickle file.
        Args:
            path (str): Path to load the preprocessor from.
        """
        with open(path, "rb") as f:
            preprocessor = pickle.load(f)
            self.numerical_scaler = preprocessor['numerical_scaler']
            self.categorical_columns = preprocessor['categorical_columns']
            self.numerical_columns = preprocessor['numerical_columns']
