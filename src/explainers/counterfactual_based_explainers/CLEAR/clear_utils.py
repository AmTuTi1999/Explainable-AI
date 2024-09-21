import numpy as np
import logging
from typing import Callable
from pandas import DataFrame
from src.functions.wachter import wachter_search
from sklearn.linear_model import LogisticRegression #type: ignore
from .surrogate_regressors.linear_regressor import LinearRegressor
from src.functions.norms import median_absolute_deviation, square_difference



class EstimatedBCounterfactual():
    def __init__(
            self,
            prediction_lower_bound: float = 0.3, 
            prediction_upper_bound: float = 0.7,
            norm_func: Callable = median_absolute_deviation,
            loss_func: Callable = square_difference,
            balanced_neighbourhood: bool = True,
            regressor_model: Callable = None,
            number_of_points_per_neighbourhood: int = 1500,
        ):
        self.p_l_b = prediction_lower_bound
        self.p_u_b = prediction_upper_bound
        self.norm_func = norm_func
        self.loss_func = loss_func
        self.balanced_neighbourhood = balanced_neighbourhood
        if regressor_model is None:
            self.regressor_model = LinearRegressor(LogisticRegression())
        else:
            self.regressor_model = LinearRegressor(regressor_model)
        self.number_of_points_per_neighbourhood = number_of_points_per_neighbourhood

    def __call__(
            self,
            x_batch,
            input_vector,
            model,
            counterfactual_target_class,
            search_space,
        ):

        estimated_b_counterfactuals, b_counterfactuals = self.explain_instance(
            x_batch, model, input_vector, counterfactual_target_class, search_space
        )
        return estimated_b_counterfactuals, b_counterfactuals

    def get_set_from_prediction_boundaries(
            self,
            x_batch,
            counterfactual_class_probabilities,
            upper_bound: float,
            lower_bound: float,
        ) -> DataFrame:

        upper_boundary_set = np.array(x_batch)[counterfactual_class_probabilities > lower_bound]
        boundary_set_array = upper_boundary_set[counterfactual_class_probabilities < upper_bound]
        boundary_set_df = DataFrame(boundary_set_array, columns=x_batch.columns)
        return boundary_set_df


    def get_input_neighbourhood(
            self,
            x_batch,
            model,
            input_vector,
            target_class,
        ) -> tuple[DataFrame, DataFrame]:

        neighbourhood = np.array([])

        prediction_probabilities = model.predict_proba(x_batch)
        counterfactual_class_probabilities = prediction_probabilities[:, target_class]

        lower_cut = self.get_set_from_prediction_boundaries(x_batch, counterfactual_class_probabilities, 0, self.p_l_b)
        mid_cut = self.get_set_from_prediction_boundaries(x_batch, counterfactual_class_probabilities, self.p_l_b, self.p_u_b)
        upper_cut = self.get_set_from_prediction_boundaries(x_batch, counterfactual_class_probabilities, self.p_u_b, 1)

        if self.balanced_neighbourhood:
            self.number_of_points_per_neighbourhood = min([len(lower_cut), len(mid_cut), len(upper_cut)])*3
            for cut in [lower_cut.to_numpy(), mid_cut.to_numpy(), upper_cut.to_numpy()]:
                distance_norms = [[self.norm_func(x_batch, input_vector, cut[i]), i] for  i in range(len(cut))]
                distance_norms = np.array(distance_norms).reshape((len(cut), 2))
                sorted_distance_norms = distance_norms[distance_norms[:,0].argsort()]
                neighbourhood_index_list = sorted_distance_norms[:,1].astype(int)[self.number_of_points_per_neighbourhood//3]
                selected_points = cut[neighbourhood_index_list]
                neighbourhood.append(selected_points)
        else:
            for cut in [lower_cut.to_numpy(), mid_cut.to_numpy(), upper_cut.to_numpy()]:
                distance_norms = [[self.norm_func(x_batch, input_vector, cut[i]), i] for  i in range(len(cut))]
                distance_norms = np.array(distance_norms).reshape((len(cut), 2))
                sorted_distance_norms = distance_norms[distance_norms[:,0].argsort()]
                selected_points = cut[neighbourhood_index_list]
                if self.number_of_points_per_neighbourhood//3 < len(cut):
                    index_list = sorted_distance_norms[:,1].astype(int)[:self.number_of_points_per_neighbourhood//3]
                    selected_points = cut[index_list]
                else:
                    selected_points = cut
                neighbourhood.append(selected_points)
                
        neighbourhood_set = np.concatenate(neighbourhood)
        neighbourhood_df = DataFrame(neighbourhood_set, columns=x_batch.columns)
        neighbourhood_labels = DataFrame(model.predict(neighbourhood_df), columns=['target'])
        return neighbourhood_df, neighbourhood_labels
    
    def find_weights(
            self,
            x_batch,
            input_vector,
            model,
            target_class,
    ):
        data, labels = self.get_input_neighbourhood(
            x_batch,
            model,
            input_vector, 
            target_class,
        )
        regressor_weights = self.regressor_model.weights(data, labels)
        return regressor_weights	


    def explain_instance(
            self,
            x_batch,
            model,
            input_vector,
            target_class,
            search_space,
        ):
        list_of_counterfactuals = wachter_search(
            input_vector, x_batch, model, search_space, self.loss_func, self.norm_func
            ).to_numpy()
        
        weights = self.find_weights(x_batch, input_vector, model, target_class)
        
        estimate = np.zeros((list_of_counterfactuals.shape))
        for i in range(estimate.shape[0]):
            for k in range(estimate.shape[1]):
                b_counter = list_of_counterfactuals[i].copy()
                b_counter[k] = 0
                estimate[i][k] = (-1*(np.multiply(weights, b_counter)).sum())/(weights[k] + 1e-8)
        estimated_b_counterfactuals = estimate
        b_counterfactuals = list_of_counterfactuals
        return estimated_b_counterfactuals, b_counterfactuals

