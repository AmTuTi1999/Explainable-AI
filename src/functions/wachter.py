import numpy as np
from pandas import DataFrame
from typing import Callable 

def wachter_search(
    input_vector: np.ndarray, 
    x_batch: np.ndarray,
    model: Callable,
    search_space: DataFrame, 
    loss_func: Callable,
    norm_func: Callable,
    ):   
    
    prediction_probabilities = model.predict_proba(search_space)
    best_possible_classification = 1.0

    distances = np.array(
        [norm_func(x_batch, search_space.iloc[i], input_vector) for i in range(len(search_space))]
    )
    losses = np.array(
        [loss_func(best_possible_classification, max(prediction_probabilities[i])) for i in range(len(search_space))]
    )
    
    sums = distances + losses
    sorted_list = sums.argsort()[::-1]
    index = sorted_list
    return DataFrame(np.array(search_space.iloc[index]), columns=search_space.columns)
    


        