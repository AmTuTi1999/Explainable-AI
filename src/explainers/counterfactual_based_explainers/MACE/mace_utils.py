"""lover """
from src.functions.norms import proximity_score
from src.explainers.counterfactual_based_explainers.MACE.init import count_differences, select_top_feature_values, select_top_features

def counterfactual_feature_selection(neighbor_data, nearest_neighbors, query_instance,  s, m):
    """
    Perform counterfactual feature selection for the given query instance.
    
    Parameters:
        data: ndarray, training data
        labels: ndarray, training labels
        query_instance: ndarray, the query instance
        predicted_label: int, predicted label of the query instance
        K: int, number of nearest neighbors
        s: int, number of selected feature columns
        m: int, number of selected values for each feature column
    
    Returns:
        Selected feature columns (C) and their corresponding selected values (V(C)).
    """
    neighbor_instances = neighbor_data[nearest_neighbors.flatten()]
    n_features = neighbor_data.shape[1]
    col_count, val_count = count_differences(query_instance, neighbor_instances, n_features)
    selected_features = select_top_features(col_count, s)
    selected_values = select_top_feature_values(val_count, selected_features, m)
    return selected_features, selected_values

def counterfactual_example_selection(E, x, K):
    """
    Select counterfactual examples based on proximity scores and feature counts.
    
    Args:
    - E: List of counterfactual examples where each example is a dictionary with keys 'categorical' and 'continuous'
    - x: Original instance as a dictionary with keys 'categorical' and 'continuous'
    - K: Threshold for feature counts
    
    Returns:
    - R: List of selected counterfactual examples
    """
    x = x.flatten()
    E_sorted = sorted(E, key=lambda x_prime: proximity_score(x, x_prime), reverse=True)
    feature_counts = {f: 0 for f in range(len(x))}
    R = []
    
    for x_prime in E_sorted:
        D_x_prime = [f for f in range(len(x)) if x_prime[f] != x[f]]
        add_to_R = False
        for f in D_x_prime:
            if feature_counts[f] < K:
                add_to_R = True
                break
        if add_to_R:
            R.append(x_prime)
            for f in D_x_prime:
                feature_counts[f] += 1
    return R