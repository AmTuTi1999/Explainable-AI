import pandas as pd
from src.explainers.counterfactual_explanation import CounterfactualExplanation
import numpy as np

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


def make_attributions_into_array(attributions: dict[str, list[CounterfactualExplanation]]):
    return {
        explainer_name: np.stack(
            [
                exp.counterfactuals[0] for exp in attributions_list
            ],
            axis=0,
        ).squeeze()
        for explainer_name, attributions_list in attributions.items()
    }

def make_x_batch_and_y_batch(
    attributions: dict[str, list[CounterfactualExplanation]]
):
    """
    Convert a dictionary of CounterfactualExplanation objects into a tuple of numpy arrays.

    Parameters:
    - attributions (dict): A dictionary where keys are explainer names and values are lists of CounterfactualExplanation objects.

    Returns:
    - tuple: A tuple containing two dictionaries:
        - x_batch: A dictionary where keys are explainer names and values are numpy arrays of input vectors.
        - y_batch: A dictionary where keys are explainer names and values are numpy arrays of target classes.
    """
    x_batch = {}
    y_batch = {}
    for explainer_name, attributions_list in attributions.items():
        x_batch[explainer_name] = np.stack(
            [
                exp.input_vector for exp in attributions_list
            ],
            axis=0,
        ).squeeze() 
        y_batch[explainer_name] = np.stack(
            [
                exp.actual_class for exp in attributions_list
            ],
            axis=0,
        ).squeeze()

    return x_batch, y_batch


def make_attrributes_individual(
    attributions: dict[str, list[CounterfactualExplanation]]
):
    """
    Convert a dictionary of CounterfactualExplanation objects into a dictionary of numpy arrays.

    Parameters:
    - attributions (dict): A dictionary where keys are explainer names and values are lists of CounterfactualExplanation objects.

    Returns:
    - dict: A dictionary where keys are explainer names and values are numpy arrays of counterfactuals.
    """
    individual_counterfactual_dict = {}
    for explainer_name, attributions_list in attributions.items():
        individual_counterfactual_list = []
        for i in range(len(attributions_list)):
            for j in range(len(attributions_list[i].counterfactuals)):
                attributions_list[i].counterfactuals[j] = np.array(attributions_list[i].counterfactuals[j])
                individual_counterfactual_list.append(CounterfactualExplanation(
                    input_vector=attributions_list[i].input_vector,
                    counterfactuals=attributions_list[i].counterfactuals[j],
                    feature_names=attributions_list[i].feature_names,
                    actual_class=attributions_list[i].actual_class,
                    counterfactual_target_class=attributions_list[i].counterfactual_target_class,
                ))
        individual_counterfactual_dict[explainer_name] = individual_counterfactual_list

    return individual_counterfactual_dict
