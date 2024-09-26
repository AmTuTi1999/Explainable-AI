"""augment data function"""
from typing import Callable
import numpy as np
import pandas as pd

from src.utils import get_categorical_and_numerical_columns
from src.functions.perturb import gaussian_2d, categorical_2d

def data_augment(
        x_batch: pd.DataFrame, 
        y_batch: pd.DataFrame,
        model: Callable,
        immutable_columns: list[str] = None,
        number_of_iterations: int = 10,
        numerical_perturb_func: Callable = gaussian_2d,
        categorical_perturb_func: Callable = categorical_2d,
    ):
    """_summary_

    Args:
        x_batch (pd.DataFrame): _description_
        y_batch (pd.DataFrame): _description_
        model (Callable): _description_
        immutable_columns (list[str], optional): _description_. Defaults to None.
        number_of_iterations (int, optional): _description_. Defaults to 10.
        numerical_perturb_func (Callable, optional): _description_. Defaults to gaussian_2d.
        categorical_perturb_func (Callable, optional): _description_. Defaults to categorical_2d.

    Returns:
        _type_: _description_
    """    
    assert(isinstance(x_batch, pd.DataFrame))

    categorical_columns, numerical_columns = get_categorical_and_numerical_columns(x_batch)
    numerical_synthetic_array = []
    categorical_synthetic_array = []
    immutable_synthetic_array = []

    synthetic_df = pd.DataFrame()
    for _ in range(number_of_iterations):
        numerical_synthetic_array.append(numerical_perturb_func(x_batch[numerical_columns]))
        categorical_synthetic_array.append(categorical_perturb_func(x_batch[categorical_columns]))
        if immutable_columns is not None:
            immutable_synthetic_array.append(np.array(x_batch[immutable_columns]))

    synthetic_df[numerical_columns] = pd.DataFrame(
        np.concatenate(numerical_synthetic_array, axis=0), columns=numerical_columns
    )
    synthetic_df[categorical_columns] = pd.DataFrame(
        np.concatenate(categorical_synthetic_array, axis=0), columns=categorical_columns
    )
    if immutable_columns is not None:
        synthetic_df[immutable_columns] = pd.DataFrame(
            np.concatenate(immutable_synthetic_array, axis=0), columns=immutable_columns
        )
    synthetic_labels = pd.DataFrame(model.predict(synthetic_df), columns= ['labels'])
    

    augmented_data, augmented_labels = pd.concat([x_batch, synthetic_df], axis=0), pd.concat([y_batch, synthetic_labels], axis=0)
    return augmented_data, augmented_labels
        

