import pandas as pd

def get_categorical_and_numerical_columns(df: pd.DataFrame):
    """
    Get lists of categorical and numerical column names from a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    tuple: A tuple containing two lists:
        - List of categorical column names.
        - List of numerical column names.
    """
    # Ensure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

    return categorical_columns, numerical_columns
