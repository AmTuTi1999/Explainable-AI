from typing import Callable
import torch
import torch.nn as nn

class PredictFunction:
    """_summary_
    """    
    def __init__(self, model: Callable) -> None:
        self.model = model
    
    def __call__(self, 
                 input_vector,
                 return_probabilities: bool = False
    ):
        if return_probabilities:   
            if isinstance(self.model, nn.Module):
                prediction = self.model(input_vector)
            else:
                prediction = self.model.predict_proba(input_vector)
        else:
            if isinstance(self.model, nn.Module):
                prediction = torch.argmax(self.model(input_vector))
            else:
                prediction = self.model.predict(input_vector)

        return prediction
    
    # TODO add asserts, check for dataframe, add lvq class, tensorflow, sklearn model class
    # this might be at the experiment level