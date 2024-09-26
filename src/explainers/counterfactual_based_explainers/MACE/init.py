import numpy as np

def count_differences(query_instance, neighbors, n_features):
    """
    Count the differing feature columns and values between the query instance and its neighbors.
    
    Parameters:
        query_instance: ndarray, the query instance
        neighbors: ndarray, the K nearest neighbors of the query
        n_features: int, number of features
    
    Returns:
        col_count: list, count of differing feature columns
        val_count: dict, count of differing feature values for each column
    """
    col_count = np.zeros(n_features)
    val_count = {i: {} for i in range(n_features)}
    for neighbor in neighbors:
        print(neighbor)
        for c in range(n_features):
            if query_instance[0][c] != neighbor[c]:
                col_count[c] += 1
                val_value = neighbor[c]
                if val_value in val_count[c]:
                    val_count[c][val_value] += 1
                else:
                    val_count[c][val_value] = 1
    return col_count, val_count

def select_top_features(col_count, s):
    """
    Select the top s feature columns based on the col_count.
    
    Parameters:
        col_count: ndarray, count of differing feature columns
        s: int, number of top features to select
    
    Returns:
        List of top s feature indices.
    """
    sorted_indices = np.argsort(-col_count)  # Sort in descending order
    return sorted_indices[:s]

def select_top_feature_values(val_count: dict, selected_features: list, m: int):
    """
    Select the top m feature values for each selected feature column.
    
    Parameters:
        val_count: dict, count of differing feature values for each column
        selected_features: list, indices of selected feature columns
        m: int, number of top feature values to select for each column
    
    Returns:
        Dictionary of selected feature values for each column.
    """
    selected_values = {}
    
    for c in selected_features:
        sorted_values = sorted(val_count[c].items(), key=lambda x: -x[1])  # Sort by count, descending
        selected_values[c] = [val for val, count in sorted_values[:m]]
    
    return selected_values

def objective_function(x, z, C):
    """
    Define the objective function for GLD method.
    
    Args:
    - x: Original instance.
    - z: Counterfactual example.
    - C: Set of modified continuous features.
    
    Returns:
    - Objective value.
    """
    print(x)
    print(z)
    return np.sum([abs(z[i] - x[0][i]) for i in C])

def gradient_less_descent(classifier, x, x_prime_list, R, r, T, counterfactual_target_class):
    """
    Optimize continuous features of counterfactual examples using GLD method.
    
    Args:
    - classifier: A function that predicts the label probability.
    - x: Original instance.
    - x_prime_list: List of counterfactual examples.
    - R: Maximum search radius.
    - r: Minimum search radius.
    - T: Number of iterations per epoch.
    
    Returns:
    - List of optimized counterfactual examples.
    """
    optimized_examples = []
    x = np.array(x)
    for x_prime in x_prime_list:
        K = int(np.log2(R / r))
        C = [i for i in range(len(x_prime)) if x_prime[i] != x.flatten()[i]]  # Set of modified continuous features
        z = x_prime.copy()
        
        X = []  
        for _ in range(T):
            T_t = []  
            for k in range(1, K + 1):
                r_k = 2 ** (-k) * R
                z_k = z.copy()
                for i in C:
                    a = np.random.normal(0, 1)  
                    z_k[i] = float(np.clip(z[i] + a * r_k, 0, 1))
                    print('ffffff')
                    print(classifier(np.array(z_k).reshape((1,-1))))
                if classifier(np.array(z_k).reshape((1,-1)))> 0.5:
                    T_t.append(z_k)
            if T_t:
                best_z = min(T_t, key=lambda z_t: objective_function(x, z_t, C))
                z = best_z
                X.append(z)
        if X:
            optimized_examples.append(min(X, key=lambda z: objective_function(x, z, C)))
        else:
            optimized_examples.append(z)
    
    return optimized_examples
