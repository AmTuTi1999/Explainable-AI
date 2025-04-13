import dice_ml
from typing import Union

import pandas as pd
import numpy as np
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.counterfactual_explanation import CounterfactualExplanation

class DICE(CounterfactualExplainerBase):
    """_summary_
    """    
    def __init__(
        self,
        num_counterfactuals: int = 4,
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
        self.num_counterfactuals = num_counterfactuals
            
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
            dice_x_batch = self.x_batch.copy()
            dice_x_batch['labels'] = self.y_batch
            # non_continuous_features = self.categorical_features + self.immutable_features 
            # continuous_features = [features for features in self.x_batch.columns if features not in non_continuous_features]
            data= dice_ml.Data(
                dataframe=dice_x_batch,
                continuous_features=self.x_batch.columns.to_list(),
                outcome_name='labels',
            )
            self.model = dice_ml.Model(model=model, backend="PYT", model_type="classifier")
            self.counterfactual_generator = dice_ml.Dice(data, self.model, method="random")
                            
    def alias(self):
        return "DICE"
    
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
        instance_class = 0
        data = pd.DataFrame(np.array(input_vector).reshape(1,-1), columns=self.x_batch.columns.to_list())
        counterfactuals = self.counterfactual_generator.generate_counterfactuals(
             data, total_CFs=self.num_counterfactuals, desired_class=counterfactual_target_class
        )
        counterfactuals = counterfactuals.cf_examples_list[0].final_cfs_df
        counterfactuals = counterfactuals.drop(columns=['labels'])
        counterfactuals = counterfactuals.to_numpy()

        return CounterfactualExplanation(
            input_vector=input_vector,
            counterfactuals=counterfactuals,
            feature_names=self.x_batch.columns.to_list(),
            actual_class=instance_class,
            counterfactual_target_class=counterfactual_target_class,
            graph=None
        )
    
        
    def transform_to_df(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return pd.DataFrame(X, columns=self.feature_names)