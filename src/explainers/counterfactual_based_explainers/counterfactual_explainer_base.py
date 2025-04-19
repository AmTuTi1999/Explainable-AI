import copy
import collections
from typing import Callable, Union
import logging

import sklearn
import torch
import pandas as pd
import numpy as np
import scipy as sp
import lime.discretize as dct
from src.explainers.helpers.helpers import extend_dataframe
from src.explainers.counterfactual_based_explainers.preprocess.data_augmentation import data_augment
from src.explainers.counterfactual_explanation import CounterfactualExplanation
from tqdm import tqdm

class CounterfactualExplainerBase():
    """_summary_
    """    
    def __init__(
        self,
        discretize_continuous: bool = False,
        discretizer: str = "decile",
        random_state: int = 42,
        data_augmentation: bool = False,
        feature_names = None,
        categorical_features: list[str] = None, 
        categorical_names = None,
        immutable_features: list[str]= None, 
    ):
        """_summary_

        Args:
            model (Callable): _description_
            x_batch (_type_): _description_
            y_batch (_type_): _description_
            categorical_features (list[str], optional): _description_. Defaults to None.
            categorical_names (_type_, optional): _description_. Defaults to None.
            immutable_features (list[str], optional): _description_. Defaults to None.
            feature_names (_type_, optional): _description_. Defaults to None.
            x_batch_stats (dict, optional): _description_. Defaults to None.
            discretize_continuous (bool, optional): _description_. Defaults to True.
            discretizer (str, optional): _description_. Defaults to "decile".
            random_state (int, optional): _description_. Defaults to 42.

        Raises:
            ValueError: _description_
        """        
        self.categorical_features = categorical_features or []
        self.immutable_features = immutable_features or []
        self.feature_names = feature_names or []
        self.discretizer = discretizer
        self.discretize_continuous = discretize_continuous
        self.random_state = random_state
        self.categorical_names= categorical_names or {},
        self.data_augmentation = data_augmentation
                    


    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]


    def _check_and_preprocess_data(
            self, 
            model: Callable,
            x_batch, 
            y_batch,
            x_batch_stats=None,
            ):
        """
            Method to preprocess the batch data
        """ 
        if isinstance(x_batch, pd.DataFrame):
            self.columns = x_batch.columns
        
        if isinstance(y_batch, pd.DataFrame):
            self.y_batch = pd.DataFrame(y_batch.values, columns=['labels'])

        if isinstance(y_batch, np.ndarray):
            self.y_batch = pd.DataFrame(y_batch, columns=['labels'])

        if isinstance(y_batch, torch.Tensor):
            self.y_batch= pd.DataFrame(y_batch.numpy(), columns= ['labels'])

        if isinstance(y_batch, pd.Series):
            self.y_batch = pd.DataFrame(y_batch.values, columns= ['labels'])
        
        if isinstance(x_batch, np.ndarray):
            self.x_batch = pd.DataFrame(x_batch)
            self.columns = list(range(x_batch.shape[1]))

        if isinstance(x_batch, torch.Tensor):
            self.x_batch = pd.DataFrame(x_batch.numpy())
            self.columns = list(range(x_batch.shape[1]))
        if self.data_augmentation:
            synthetic_data, synthetic_labels = data_augment(
                self.x_batch, self.y_batch, model, immutable_columns=self.immutable_features
                )
            self.x_batch, self.y_batch = extend_dataframe(self.x_batch, synthetic_data), extend_dataframe(self.y_batch, synthetic_labels)
        if x_batch_stats:
            self.validate_x_batch_stats(x_batch_stats)

        if self.categorical_features is None:
            self.categorical_features = []
        if self.feature_names is None:
            self.feature_names = [str(i) for i in range(x_batch.shape[1])]

        self.categorical_features = list([int(i) for i in range(len(self.categorical_features))])
        self.feature_names = list(self.columns)

        self.discretizer = None
        if self.discretize_continuous and not sp.sparse.issparse(self.x_batch):
            # Set the discretizer if training data stats are provided
            if x_batch_stats:
                discretizer = dct.StatsDiscretizer(
                    self.x_batch, self.categorical_features,
                    self.feature_names, labels=self.y_batch,
                    data_stats=x_batch_stats,
                    random_state=self.random_state)

            if discretizer == 'quartile':
                self.discretizer = dct.QuartileDiscretizer(
                        self.x_batch.to_numpy(), self.categorical_features,
                        self.feature_names, labels=self.y_batch,
                        random_state=self.random_state)
            elif discretizer == 'decile':
                self.discretizer = dct.DecileDiscretizer(
                        self.x_batch.to_numpy(), self.categorical_features,
                        self.feature_names, labels=self.y_batch,
                        random_state=self.random_state)
            elif discretizer == 'entropy':
                self.discretizer = dct.EntropyDiscretizer(
                        self.x_batch.to_numpy(), self.categorical_features,
                        self.feature_names, labels=self.y_batch,
                        random_state=self.random_state)
            elif isinstance(discretizer, dct.BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(self.x_batch.shape[1]))

            # Get the discretized_x_batch when the stats are not provided
            discretized_x_batch = self.discretizer.discretize(
                np.array(self.x_batch))
            self.x_batch = pd.DataFrame(discretized_x_batch, columns=self.columns)
        self.feature_values = {}
        self.feature_frequencies = {}
        for feature in self.categorical_features:
            if x_batch_stats is None:
                if self.discretizer is not None:
                    column = discretized_x_batch[:, feature]
                else:
                    column = self.x_batch[feature]

                feature_count = collections.Counter(column)
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = x_batch_stats["feature_values"][feature]
                frequencies = x_batch_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))

    @staticmethod
    def validate_x_batch_stats(x_batch_stats):
        """
            Method to validate the structure of training data stats
        """
        stat_keys = list(x_batch_stats.keys())
        valid_stat_keys = ["means", "mins", "maxs", "stds", "feature_values", "feature_frequencies"]
        missing_keys = list(set(valid_stat_keys) - set(stat_keys))
        if len(missing_keys) > 0:
            raise ValueError("Missing keys in x_batch_stats. Details: %s" % (missing_keys))
        
    def __call__(self, *args, **kwds):
        pass

    def _init_explainer(self, model, x_batch, y_batch, x_batch_stats=None):
        """_summary_

        Args:
            x_batch (_type_): _description_
            y_batch (_type_): _description_
        """        
        self.model = model
        self._check_and_preprocess_data(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            x_batch_stats=x_batch_stats,
        )

    def init_explainer(
            self,
            model: Callable,
            x_batch: pd.DataFrame,
            y_batch: pd.DataFrame,
            x_batch_stats: dict = None,
    ):
        pass

    def explain_instance(
            self,
            input_vector,
            counterfactual_target_class: Union[int, str] = "opposite",
        ) -> CounterfactualExplanation:
        pass 
    
    def explainer_first_step(self, input_vector: np.ndarray):
        """_summary_

        Args:
            input_vector (np.ndarray): _description_

        Returns:
            _type_: _description_
        """        
        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(input_vector.shape[0])]

        if sp.sparse.issparse(input_vector):
            values = self.convert_and_round(input_vector.data)
            #feature_indexes = input_vector.indices
        else:
            values = self.convert_and_round(input_vector)
            #feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(input_vector[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        #categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretize_continuous and self.discretizer is not None:
            #categorical_features = x_batch.columns
            discretized_instance = self.discretizer.discretize(np.array(input_vector))
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]
            return discretized_instance
        else:
            return input_vector
    
    def explain_batch(
            self,
            data,
            num_explanations,
            counterfactual_target_class,
        ) -> list[CounterfactualExplanation]:
        """_summary_

        Args:
            num_explanations (_type_): _description_
            counterfactual_target_class (_type_): _description_
        """ 
        if data.ndim == 3:
            pd_data = pd.DataFrame(data.squeeze(1).numpy())
        else:
            pd_data = pd.DataFrame(data.numpy())
        counterfactual_explanation_list = []    
        if len(pd_data) == 1:
            counterfactual_explanation_list.append(self.explain_instance(pd_data.iloc[0], counterfactual_target_class))
        elif len(pd_data) > 1:
            for i in tqdm(range(num_explanations), desc="Generating counterfactual explanations"):
                counterfactual_explanation_list.append(self.explain_instance(pd_data.iloc[i], counterfactual_target_class))
        return counterfactual_explanation_list
    
    def alias(self):
        pass
    