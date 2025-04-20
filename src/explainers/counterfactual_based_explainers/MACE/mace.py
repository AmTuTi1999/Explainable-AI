"""MACE"""
import logging
from typing import Callable, Union

import pandas as pd
import numpy as np
from src.explainers.helpers.helpers import get_opposite_class
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.counterfactual_based_explainers.MACE.knn import build_knn_tree, find_k_nearest_neighbors
from src.explainers.counterfactual_based_explainers.MACE.create_env import BuildModelEnv
from src.explainers.counterfactual_based_explainers.MACE.init import gradient_less_descent
from src.explainers.counterfactual_based_explainers.MACE.reinforce import rl_based_counterfactual_optimization
from src.explainers.counterfactual_based_explainers.MACE.mace_utils import counterfactual_feature_selection, counterfactual_example_selection
from src.explainers.counterfactual_explanation import CounterfactualExplanation

class MACE(CounterfactualExplainerBase):
    """_summary_
    """    
    def __init__(
        self,
        top_num_features: int,
        top_num_feature_values: int,
        num_points_neighbourhood: int,
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
        discretizer: str = 'decile',
        feature_names = None,
        categorical_features: list[str] = None, 
        immutable_features: list[str]= None, 
    ):
        super().__init__(
            immutable_features=immutable_features,
            categorical_features = categorical_features,
            feature_names = feature_names,
            discretize_continuous=discretize_continuous,
            discretizer=discretizer
    )
        self.num_points_neighbourhood = num_points_neighbourhood
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
            
    def __call__(
            self, 
            num_explanations,
            counterfactual_target_class,
        ):
        self.explain_batch(
            num_explanations, counterfactual_target_class
            )
    def init_explainer(
            self,
            model,
            x_batch,
            y_batch,
            x_batch_stats = None,
        ): 
            self._init_explainer(                                                      
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                x_batch_stats=x_batch_stats, 
            )     
                            
    def alias(self):
        return "MACE"
    
    def explain_instance(
            self,
            input_vector,
            counterfactual_target_class: Union[int, str] = "opposite",
        ) -> CounterfactualExplanation:
        """_summary_

        Args:
            input_vector (_type_): _description_
            counterfactual_target_class (int or str, optional): _description_. Defaults to "opposite".
        """      
        instance_class = self.model.predict(input_vector.to_frame().T)
        input_vector = self.explainer_first_step(input_vector)
        input_vector = pd.DataFrame(np.array(input_vector).reshape((1,-1)), columns=self.feature_names)
        if counterfactual_target_class == 'opposite':
            counterfactual_target_class = get_opposite_class(instance_class)

        knn_tree, neighbor_data = build_knn_tree(
            self.x_batch, self.y_batch, counterfactual_target_class, self.immutable_features, self.num_points_neighbourhood
            )
        nearest_neighbours, _ = find_k_nearest_neighbors(neighbor_data, knn_tree, input_vector, 10, self.immutable_features) # TODO change magic number
        _ , selected_feature_values = counterfactual_feature_selection(neighbor_data.to_numpy(), nearest_neighbours, input_vector.to_numpy(), self.s, self.m)
        model_env = BuildModelEnv(self.model, counterfactual_target_class, input_vector.to_numpy(), self.discretizer)
        counterfactual_examples = rl_based_counterfactual_optimization(
             model_env, input_vector.to_numpy(), selected_feature_values, self.w, self.gamma, self.num_episodes, self.alpha, self.lambda1, self.lambda2
        )
        selected_counterfactual_examples = counterfactual_example_selection(counterfactual_examples, input_vector.to_numpy(), self.m)
        refined_counterfactuals =  gradient_less_descent(
            model_env.predict, input_vector.to_numpy(), selected_counterfactual_examples, self.max_search_radius, self.min_search_radius, self.refine_epochs, counterfactual_target_class
        )[:self.b]
        counterfactual_predictions = self.model.predict(np.array(refined_counterfactuals))
        print(len(np.array(refined_counterfactuals)))
        counterfactual_probabilities = np.max(self.model.predict_proba(np.array(refined_counterfactuals)), axis=1)
        print(f"Counterfactual probabilities: {counterfactual_probabilities}")
        # TODO add representation function, returns dataframe, html, or something else: do research

        return CounterfactualExplanation(
            input_vector=input_vector,
            counterfactuals=np.array(refined_counterfactuals),
            feature_names=self.feature_names,
            actual_class=instance_class,
            counterfactual_target_class=counterfactual_target_class,
            graph=None,
            counterfactual_predictions=counterfactual_predictions,
            counterfactual_probabilities=counterfactual_probabilities,
        )
    
        
    def transform_to_df(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return pd.DataFrame(X, columns=self.feature_names)