import numpy as np
import pandas as pd
import logging
from typing import Callable
from explainers.helpers.helpers import extend_dataframe, get_opposite_class

from explainers.counterfactual_based_explainers.preprocess.data_augmentation import data_augment
from src.explainers.counterfactual_based_explainers.CLEAR.surrogate_regressors.linear_regressor import LinearRegressor
from src.explainers.counterfactual_based_explainers.CLEAR.clear_utils import EstimatedBCounterfactual

class CLEAR:
    def __init__(
        self,
        model, 
        x_batch,
        y_batch,
        num_points_neighbourhood,
        immutable_features: list[str] = [],  
        regressor_model: Callable = None,    
        ):
            self.num_points_neighbourhood = num_points_neighbourhood
            self.model = model
            self.regressor_model = regressor_model
            self.x_batch = x_batch
            self.columns = x_batch.columns
            synthetic_data, synthetic_labels = data_augment(
                x_batch, y_batch, model, immutable_columns=immutable_features
                )
            self.augmented_data, self.augmented_labels = extend_dataframe(x_batch, synthetic_data), extend_dataframe(y_batch, synthetic_labels)

    def __call__(
            self, 
            num_explanations,
            counterfactual_target_class,
        ):
        self.explain_batch(num_explanations, counterfactual_target_class)

    def explain_instance(
            self,
            input_vector,
            counterfactual_target_class: int | str = "opposite",
        ):

        search_space = self.augmented_data[self.augmented_labels['label'] == counterfactual_target_class]        
        instance_class = self.model.predict(input_vector)

        if counterfactual_target_class == 'opposite':
            logging.info("Calling Explainer for Binary Class")
            counterfactual_target_class = get_opposite_class(instance_class)

        estimated_b_counterfactuals, b_counterfactuals = EstimatedBCounterfactual(
            regressor_model=self.regressor_model, 
            number_of_points_per_neighbourhood=self.num_points_neighbourhood
            )(
            input_vector, self.model, counterfactual_target_class, search_space
            )
        
        best_fidelity_error = np.inf
        for i in range(len(b_counterfactuals)):
            fidelity_error = fidelity_error(estimated_b_counterfactuals.iloc[i], b_counterfactuals.iloc[i], input_vector)
            if fidelity_error < best_fidelity_error:
                best_b_counterfactual = b_counterfactuals.iloc[i]
                best_estimated_b_counterfactual = estimated_b_counterfactuals.iloc[i]
                best_fidelity_error = fidelity_error
        return best_b_counterfactual, best_estimated_b_counterfactual
 
    def explain_batch(
            self,
            num_explanations,
            counterfactual_target_class,
        ):
        for i in range(num_explanations):
            self.explain_instance(self.x_batch.iloc[i], counterfactual_target_class)
    
        
    def transform_to_df(self, X):
        return pd.DataFrame(X, columns=self.columns)


    
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
    
        


    