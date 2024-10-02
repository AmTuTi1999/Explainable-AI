import copy
import collections
from typing import Callable

import sklearn
import torch
import pandas as pd
import numpy as np
import scipy as sp
import lime.discretize as dct
from src.explainers.helpers.helpers import extend_dataframe
from src.explainers.counterfactual_based_explainers.preprocess.data_augmentation import data_augment

class CounterfactualExplainerBase():
    """_summary_
    """    
    def __init__(
        self,
        model: Callable, 
        x_batch,
        y_batch,
        categorical_features: list[str] = None, 
        categorical_names = None,
        immutable_features: list[str]= None, 
        feature_names = None,
        x_batch_stats: dict = None,
        discretize_continuous: bool = False,
        discretizer: str = "decile",
        random_state: int = 42,
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
        self.model = model
        self.discretize_continuous = discretize_continuous
        self.x_batch_stats = x_batch_stats
        self.random_state = random_state
        self.categorical_names= categorical_names or {},
                    

        if isinstance(x_batch, pd.DataFrame):
            self.columns = x_batch.columns
        
        if isinstance(y_batch, pd.DataFrame):
            y_batch = pd.DataFrame(y_batch.values, columns=['labels'])

        if isinstance(y_batch, np.ndarray):
            y_batch = pd.DataFrame(y_batch, columns=['labels'])

        if isinstance(y_batch, torch.Tensor):
            y_batch= pd.DataFrame(y_batch.numpy(), columns= ['labels'])

        if isinstance(y_batch, pd.Series):
            y_batch = pd.DataFrame(y_batch.values, columns= ['labels'])
        
        if isinstance(x_batch, np.ndarray):
            x_batch = pd.DataFrame(x_batch)
            self.columns = list(range(x_batch.shape[1]))

        if isinstance(x_batch, torch.Tensor):
            x_batch = pd.DataFrame(x_batch.numpy())
            self.columns = list(range(x_batch.shape[1]))

        synthetic_data, synthetic_labels = data_augment(
            x_batch, y_batch, model, immutable_columns=immutable_features
            )
        self.x_batch, self.y_batch = extend_dataframe(x_batch, synthetic_data), extend_dataframe(y_batch, synthetic_labels)
        print(f'x_batch: {len(self.x_batch)}')
        print(f'y_batch: {len(self.y_batch)}')
        if self.x_batch_stats:
            self.validate_x_batch_stats(self.x_batch_stats)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(x_batch.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(self.x_batch):
            # Set the discretizer if training data stats are provided
            if self.x_batch_stats:
                discretizer = dct.StatsDiscretizer(
                    self.x_batch, self.categorical_features,
                    self.feature_names, labels=y_batch,
                    data_stats=self.x_batch_stats,
                    random_state=self.random_state)

            if discretizer == 'quartile':
                self.discretizer = dct.QuartileDiscretizer(
                        self.x_batch.to_numpy(), self.categorical_features,
                        self.feature_names, labels=y_batch,
                        random_state=self.random_state)
            elif discretizer == 'decile':
                self.discretizer = dct.DecileDiscretizer(
                        self.x_batch.to_numpy(), self.categorical_features,
                        self.feature_names, labels=y_batch,
                        random_state=self.random_state)
            elif discretizer == 'entropy':
                self.discretizer = dct.EntropyDiscretizer(
                        self.x_batch.to_numpy(), self.categorical_features,
                        self.feature_names, labels=y_batch,
                        random_state=self.random_state)
            elif isinstance(discretizer, dct.BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(x_batch.shape[1]))

            # Get the discretized_x_batch when the stats are not provided
            discretized_x_batch = self.discretizer.discretize(
                np.array(self.x_batch))
            self.x_batch = pd.DataFrame(discretized_x_batch, columns=self.columns)

        # Though set has no role to play if training data stats are provided
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(self.x_batch)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if x_batch_stats is None:
                if self.discretizer is not None:
                    column = discretized_x_batch[:, feature]
                else:
                    column = self.x_batch[:, feature]

                feature_count = collections.Counter(column)
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = x_batch_stats["feature_values"][feature]
                frequencies = x_batch_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1


    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

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
            #categorical_features = self.x_batch.columns
            discretized_instance = self.discretizer.discretize(np.array(input_vector))
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]
            return discretized_instance
        else:
            return input_vector
    
    def explain_batch(self, num_explanations, counterfactual_target_class):
        pass
    