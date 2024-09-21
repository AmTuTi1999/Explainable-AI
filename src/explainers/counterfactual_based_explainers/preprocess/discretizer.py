import collections
import scipy as sp
import numpy as np
import lime.discretize as dct
from lime.discretize import BaseDiscretizer

def discretize(
        training_data, 
        training_data_stats, 
        feature_names, 
        categorical_features: list[str],
        training_labels = None,
        discretizer: str = 'quartile', 
        random_state = 42
):
    if sp.sparse.issparse(training_data):
        # Set the discretizer if training data stats are provided
        if training_data_stats:
            discretizer_func = dct.StatsDiscretizer(
                training_data, categorical_features,
                feature_names, labels=training_labels,
                data_stats=training_data_stats,
                random_state=random_state)

        if discretizer == 'quartile':
            discretizer_func = dct.QuartileDiscretizer(
                    training_data, categorical_features,
                    feature_names, labels=training_labels,
                    random_state=random_state)
        elif discretizer == 'decile':
            discretizer_func = dct.DecileDiscretizer(
                    training_data, categorical_features,
                    feature_names, labels=training_labels,
                    random_state=random_state)
        elif discretizer == 'entropy':
            discretizer_func = dct.EntropyDiscretizer(
                    training_data, categorical_features,
                    feature_names, labels=training_labels,
                    random_state=random_state)
        elif isinstance(discretizer, BaseDiscretizer):
            discretizer_func = discretizer
        else:
            raise ValueError('''Discretizer must be 'quartile',''' +
                                ''' 'decile', 'entropy' or a''' +
                                ''' BaseDiscretizer instance''')
        categorical_features = list(range(training_data.shape[1]))

        # Get the discretized_training_data when the stats are not provided
        if(training_data_stats is None):
            discretized_training_data = discretizer_func.discretize(
                training_data)
            
    feature_values = {}
    feature_frequencies = {}
    for feature in categorical_features:
        if training_data_stats is None:
            if discretizer is not None:
                column = discretized_training_data[:, feature]
            else:
                column = training_data[:, feature]

            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
        else:
            values = training_data_stats["feature_values"][feature]
            frequencies = training_data_stats["feature_frequencies"][feature]

        feature_values[feature] = values
        feature_frequencies[feature] = (np.array(frequencies) /
                                        float(sum(frequencies)))

