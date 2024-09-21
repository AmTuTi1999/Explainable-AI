import copy
import collections
from typing import Callable

import sklearn
import pandas as pd
import numpy as np
import scipy as sp
import lime.discretize as dct

class CounterfactualExplainerBase():
    def __init__(
        self,
        model, 
        x_batch,
        y_batch,
        top_num_features,
        top_num_feature_values,
        num_points_neighbourhood,
        categorical_features: list[str] = None, 
        categorical_names = None,
        immutable_features: list[str]= None, 
        feature_names = None,
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
        x_batch_stats: dict = None,
        discretize_continuous: bool = True,
        discretizer: str = "decile",
        random_state: int = 42
    ):
        self.num_points_neighbourhood = num_points_neighbourhood
        self.model = model
        self.regressor_model = regressor_model
        self.x_batch = x_batch # TODO add instances for other datatypes
        self.y_batch = y_batch
        self.columns = x_batch.columns #TODO  cut this out
        self.s = top_num_features
        self.m = top_num_feature_values
        self.gamma = gamma
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.lambda1, self.lambda2 = lambdas
        self.w = sparsity_constraint
        self.b = num_counterfactuals
        self.x_batch_stats = x_batch_stats
        self.max_search_radius = max_search_radius
        self.min_search_radius = min_search_radius
        self.refine_epochs = refine_epochs
        self.random_state = random_state
        self.categorical_names= categorical_names or {},

        # synthetic_data, synthetic_labels = data_augment(
        #     x_batch, y_batch, model, immutable_columns=immutable_features
        #     )
        # self.augmented_data, self.augmented_labels = extend_dataframe(x_batch, synthetic_data), extend_dataframe(y_batch, synthetic_labels)
        
        if self.x_batch_stats:
            self.validate_x_batch_stats(self.x_batch_stats)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(x_batch.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(x_batch):
            # Set the discretizer if training data stats are provided
            if self.x_batch_stats:
                discretizer = dct.StatsDiscretizer(
                    x_batch, self.categorical_features,
                    self.feature_names, labels=y_batch,
                    data_stats=self.x_batch_stats,
                    random_state=self.random_state)

            if discretizer == 'quartile':
                self.discretizer = dct.QuartileDiscretizer(
                        x_batch, self.categorical_features,
                        self.feature_names, labels=y_batch,
                        random_state=self.random_state)
            elif discretizer == 'decile':
                self.discretizer = dct.DecileDiscretizer(
                        x_batch.to_numpy(), self.categorical_features,
                        self.feature_names, labels=y_batch,
                        random_state=self.random_state)
            elif discretizer == 'entropy':
                self.discretizer = dct.EntropyDiscretizer(
                        x_batch, self.categorical_features,
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
            if(self.x_batch_stats is None):
                discretized_x_batch = self.discretizer.discretize(
                    np.array(x_batch))

        # if kernel_width is None:
        #     kernel_width = np.sqrt(x_batch.shape[1]) * .75
        # kernel_width = float(kernel_width)

        # if kernel is None:
        #     def kernel(d, kernel_width):
        #         return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        # kernel_fn = partial(kernel, kernel_width=kernel_width)

        # self.feature_selection = feature_selection
        # self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        # self.class_names = class_names

        # Though set has no role to play if training data stats are provided
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(x_batch)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if x_batch_stats is None:
                if self.discretizer is not None:
                    column = discretized_x_batch[:, feature]
                else:
                    column = x_batch[:, feature]

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
            raise Exception("Missing keys in x_batch_stats. Details: %s" % (missing_keys))
        
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
            feature_indexes = input_vector.indices
        else:
            values = self.convert_and_round(input_vector)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(input_vector[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(self.x_batch.shape[1])
            discretized_instance = self.discretizer.discretize(np.array(input_vector))
            print(discretized_instance)
            print(self.discretizer.undiscretize(discretized_instance))
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]
        return discretized_instance, (categorical_features, discretized_feature_names, feature_indexes)
    
    def explain_batch(self):
        pass
    