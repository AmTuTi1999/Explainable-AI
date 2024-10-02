"""MACE"""
import logging
from typing import Callable 

import pandas as pd
from src.explainers.helpers.helpers import get_opposite_class
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.counterfactual_based_explainers.MACE.knn import build_knn_tree, find_k_nearest_neighbors
from src.explainers.counterfactual_based_explainers.MACE.create_env import BuildModelEnv
from src.explainers.counterfactual_based_explainers.MACE.init import gradient_less_descent
from src.explainers.counterfactual_based_explainers.MACE.reinforce import rl_based_counterfactual_optimization
from src.explainers.counterfactual_based_explainers.MACE.mace_utils import counterfactual_feature_selection, counterfactual_example_selection

class MACE(CounterfactualExplainerBase):
    """_summary_
    """    
    def __init__(
        self,
        model, 
        x_batch,
        y_batch,
        top_num_features: int,
        top_num_feature_values: int,
        num_points_neighbourhood: int,
        categorical_features: list[str] = None,
        feature_names: list[str] = None,
        immutable_features: list[str]= None,  
        regressor_model: Callable = None,   
        gamma: float = 0.1,
        alpha: float = 0.1,
        num_episodes: int = 100,
        lambdas: tuple = (0.1, 0.1),
        sparsity_constraint: int =  15,
        num_counterfactuals: int = 5,
        max_search_radius: float = 0.1, 
        min_search_radius: float = 0.01, 
        refine_epochs: int = 10,
        discretize_continuous = True,
        discretizer: str = 'decile'
    ):
        super().__init__(
            model=model, 
            x_batch=x_batch,
            y_batch=y_batch,
            immutable_features=immutable_features,
            categorical_features = categorical_features,
            feature_names = feature_names,
            discretize_continuous=discretize_continuous,
            discretizer=discretizer
    )
        self.num_points_neighbourhood = num_points_neighbourhood
        self.model = model
        self.regressor_model = regressor_model
        self.columns = x_batch.columns
        self.s = top_num_features
        self.m = top_num_feature_values
        self.gamma = gamma
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.lambda1, self.lambda2 = lambdas
        self.w = sparsity_constraint
        self.b = num_counterfactuals
        self.max_search_radius = max_search_radius
        self.min_search_radius = min_search_radius
        self.refine_epochs = refine_epochs
        self.immutable_features = immutable_features
            
    def __call__(
            self, 
            num_explanations,
            counterfactual_target_class,
        ):
        self.explain_batch(
            num_explanations, counterfactual_target_class
            )

    def explain_instance(
            self,
            input_vector,
            counterfactual_target_class: int | str = "opposite",
        ):
        """_summary_

        Args:
            input_vector (_type_): _description_
            counterfactual_target_class (int | str, optional): _description_. Defaults to "opposite".
        """      
        instance_class = self.model.predict(input_vector.to_frame().T)
        input_vector = self.explainer_first_step(input_vector)
        input_vector = pd.DataFrame(input_vector.reshape((1,-1)), columns=self.columns)
        if counterfactual_target_class == 'opposite':
            logging.info("Calling Explainer for Binary Class")
            counterfactual_target_class = get_opposite_class(instance_class)
        knn_tree, neighbor_data = build_knn_tree(
            self.x_batch, self.y_batch, counterfactual_target_class, self.immutable_features, self.num_points_neighbourhood
            )
        nearest_neighbours, _ = find_k_nearest_neighbors(neighbor_data, knn_tree, input_vector, 10, self.immutable_features) # TODO change magic number
        _ , selected_feature_values = counterfactual_feature_selection(neighbor_data.to_numpy(), nearest_neighbours, input_vector.to_numpy(), self.s, self.m)
        model_env = BuildModelEnv(self.model, counterfactual_target_class, input_vector.to_numpy(), self.discretizer)
        #print(selected_feature_values)
        counterfactual_examples = rl_based_counterfactual_optimization(
             model_env, input_vector.to_numpy(), selected_feature_values, self.w, self.gamma, self.num_episodes, self.alpha, self.lambda1, self.lambda2
        )
        selected_counterfactual_examples = counterfactual_example_selection(counterfactual_examples, input_vector.to_numpy(), self.m)
        print(selected_counterfactual_examples)
        refined_counterfactuals =  gradient_less_descent(
            model_env.predict, input_vector.to_numpy(), selected_counterfactual_examples, self.max_search_radius, self.min_search_radius, self.refine_epochs, counterfactual_target_class
        )[:self.b]

        # TODO add representation function, returns dataframe, html, or something else: do research

        return refined_counterfactuals
    

 
    def explain_batch(
            self,
            num_explanations,
            counterfactual_target_class,
        ):
        """_summary_

        Args:
            num_explanations (_type_): _description_
            counterfactual_target_class (_type_): _description_
        """        
        for i in range(num_explanations):
            self.explain_instance(self.x_batch.iloc[i], counterfactual_target_class)
    
        
    def transform_to_df(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return pd.DataFrame(X, columns=self.columns)