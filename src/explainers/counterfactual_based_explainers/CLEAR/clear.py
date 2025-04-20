import numpy as np
import pandas as pd
import logging
from typing import Callable, Union
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.helpers.helpers import get_opposite_class

from src.explainers.counterfactual_based_explainers.CLEAR.linear_regressor import LinearRegressor
from src.explainers.counterfactual_based_explainers.CLEAR.clear_utils import EstimatedBCounterfactual
from src.explainers.counterfactual_explanation import CounterfactualExplanation

class CLEAR(CounterfactualExplainerBase):
    """CLEAR: Counterfactual Local Explanations with Adversarial Regressor"""
    def __init__(
        self,
        num_points_neighbourhood,
        discretize_continuous: bool = False,
        discretizer: str = "decile",
        random_state: int = 42,
        data_augmentation: bool = True,  
        regressor_model: Callable = None,  
        feature_names = None,
        categorical_features: list[str] = None, 
        categorical_names = None,
        immutable_features: list[str]= None,   
        ):
            super().__init__(
                categorical_features=categorical_features,
                categorical_names=categorical_names,
                immutable_features=immutable_features,
                feature_names=feature_names,
                discretize_continuous=discretize_continuous,
                discretizer=discretizer,
                random_state=random_state,
                data_augmentation=data_augmentation,
            )
            self.num_points_neighbourhood = num_points_neighbourhood
            self.regressor_model = regressor_model

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
        return 'CLEAR'
    
    def explain_instance(
            self,
            input_vector,
            counterfactual_target_class: Union[int, str] = "opposite",
        ) -> CounterfactualExplanation:
              
        instance_class = self.model.predict(input_vector)
        if counterfactual_target_class == 'opposite':
            logging.info("Calling Explainer for Binary Class")
            counterfactual_target_class = get_opposite_class(instance_class)
        search_space = self.x_batch[self.y_batch['labels'] == counterfactual_target_class[0]]
        estimated_b_counterfactuals, b_counterfactuals = EstimatedBCounterfactual(
            regressor_model=self.regressor_model, 
            number_of_points_per_neighbourhood=self.num_points_neighbourhood
            )(
            self.x_batch, input_vector, self.model, counterfactual_target_class, search_space
            )
        
        best_fidelity_error = np.inf
        
        for i in range(len(b_counterfactuals)):
            fidelity_error_value = self.fidelity_error(estimated_b_counterfactuals[i], b_counterfactuals[i], input_vector)
            if fidelity_error_value < best_fidelity_error:
                best_b_counterfactual = b_counterfactuals[i]
                best_estimated_b_counterfactual = estimated_b_counterfactuals[i]
                best_fidelity_error = fidelity_error_value
        counterfactual_predictions = self.model.predict(best_estimated_b_counterfactual)
        return CounterfactualExplanation(
            input_vector=input_vector,
            counterfactuals=best_estimated_b_counterfactual,
            feature_names=self.feature_names,
            actual_class=instance_class,
            counterfactual_target_class=counterfactual_target_class,
            counterfactual_predictions=counterfactual_predictions,
        )
    
        
    def transform_to_df(self, X):
        return pd.DataFrame(X, columns=self.feature_names)


    def fidelity_error(self, a, b, c):
        return np.linalg.norm(abs(a - c) - abs(b - c), ord=1)


    
    # def _density(self, counterfactual, target):
        
    #     if self.d_e == 'KDE':
    #         return self.kern.kde_density(np.array(counterfactual))
    #     elif self.d_e == 'KNN':
    #         return self.kern.knn_density(np.array(counterfactual))
            


    # def check_counterfactual(self, counterfactual, target):
    #     if self.backend == 'lvq':
    #         if self.model.proba_predict(counterfactual)[target] < self.c_t:
    #             if self._density(counterfactual, target) > self.d_t:
    #                 return True
    #     elif self.backend == 'sklearn':
    #         if self.model.predict_proba(counterfactual)[0][target] > self.c_t:
    #             if self._density(counterfactual, target) > self.d_t:
    #                 return True
                
    
    # def generate_counterfactual(self,unit, target_class = 'opposite'):
    #     if self.backend == 'lvq':
    #         self.unit_class = self.model.predict(unit)
    #     elif self.backend == 'sklearn':
    #         #print(pd.DataFrame(np.array(unit).reshape((1, self.train_data.shape[1])), columns=self.train_data.columns))
    #         self.unit_class = self.model.predict(pd.DataFrame(np.array(unit).reshape((1, self.train_data.shape[1])), columns=self.train_data.columns))
            

        
    #     # unit_class = self.model.predict(unit, self.prototypes, proto_labels)
    #     #counterfactual_list = self.wachter_search(unit, self.target_class)
    #     estimations, counterfactual_list = self.estimated_b_counterfactual(unit,  self.target_class)


    #     #chosen = pd.DataFrame(best_CFEs, train_data.columns)
    #     indices = []
    #     for i in range(chosen.shape[0]):
    #         if self.backend =='lvq':

    #             if self.check_counterfactual(chosen.iloc[i], self.target_class) == True:# and self.check_counterfactual(chosen_estimates[i], self.target_class) == True:
    #                 num_features = 1
    #                 while num_features <= self.train_data.shape[1] + 1:
    #                     if check_sparsity(num_features).is_sparse(unit, chosen.iloc[i]):
    #                         indices.append(i)
    #                         #break
    #                     num_features += 1            

    #         elif self.backend == 'sklearn':
    #             if self.check_counterfactual(pd.DataFrame(np.array(chosen)[i].reshape((1, self.train_data.shape[1])), columns = self.train_data.columns), self.target_class) == True \
    #                 and self.check_counterfactual(pd.DataFrame(np.array(chosen_estimates)[i].reshape((1, self.train_data.shape[1])), columns = self.train_data.columns), self.target_class) == True:
    #                 num_features = 1
    #                 while num_features <= self.train_data.shape[1]+1:
    #                     if check_sparsity(num_features).is_sparse(unit, chosen.iloc[i]):
    #                         indices.append(i)
                            
    #                         break
    #                     num_features += 1
    #     if indices == []:
    #         print('something went wrong, repeat')
    #     else:
    #         norms = np.array([self.MAD(self.train_data[self.training_labels['labels'] == self.target_class], unit, np.array(chosen.iloc[i])) for i in indices])
    #         min_norm_index = np.argmin(norms)
    #         cf_index = indices[min_norm_index]
    #         return chosen.iloc[cf_index:cf_index+1], pd.DataFrame(chosen_estimates[cf_index].reshape((1, self.train_data.shape[1])), columns=self.train_data.columns)
    
        


    