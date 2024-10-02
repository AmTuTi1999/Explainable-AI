"""building model env"""

from typing import Callable
import numpy as np

class BuildModelEnv:
    """_summary_
    """    
    def __init__(self, model: Callable, counterfactual_class: int, input_vector, discretizer):
        """_summary_

        Args:
            model (Callable): classifier (or regressor)
        """        
        self.model = model
        self.input_vector = input_vector
        self.counterfactual_class = counterfactual_class
        self.discretizer = discretizer
    
    def reset(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        return self.input_vector
    
    def predict(self, x: np.ndarray):
        """
        predict
        """  
        x = self.discretizer.undiscretize(x)
        x =x.reshape(1,-1)
        return self.model.predict_proba(x)[0][self.counterfactual_class]
    
    def step(self, x_prime):
        """_summary_

        Args:
            x_prime (_type_): _description_

        Returns:
            _type_: _description_
        """        
        reward = self.predict(x_prime)
        next_state = x_prime
        done = reward > 0.60
        print(done)
        return next_state, reward, done, {}
    