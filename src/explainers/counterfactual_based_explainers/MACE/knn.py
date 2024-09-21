import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors #type: ignore

def build_knn_tree(
        data: pd.DataFrame, 
        labels: pd.Series, 
        counterfactual_class: int, 
        immutable_columns: list,
        n_neighbors: int = 5
    ):
    """
    Build a kNN tree for instances with a specific label, accepting pandas DataFrames.
    
    Parameters:
        data: pd.DataFrame, feature matrix
        labels: pd.Series, corresponding labels for each sample
        counterfactual_class: int, label (0 or 1) for which to build the tree
        n_neighbors: int, number of neighbors for kNN
        
    Returns:
        kNN model fitted for the subset of data with the given label and subset of data itself (excluding immutable features).
    """
    # Ensure index alignment between data and labels
    labels = labels.reindex(data.index)
    
    subset_data = data.loc[labels['labels'] == counterfactual_class[0]].reset_index(drop = True)
    non_immutable_columns = [col for col in data.columns if col not in immutable_columns]
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(subset_data[non_immutable_columns].values)

    # Select the subset of data where labels match the counterfactual_class
    print(labels.columns)
    print(subset_data.index)

    return knn_model, subset_data

def find_k_nearest_neighbors(data, knn_tree, query_instance: pd.Series, k: int, immutable_columns: str):
    """
    Find K nearest neighbors for the given query instance using the provided kNN tree,
    while excluding immutable features from the search space.
    
    Parameters:
        knn_tree: trained kNN model
        query_instance: pd.Series, the instance for which to find the neighbors
        immutable_columns: list, list of column names for immutable features
        K: int, number of nearest neighbors
    
    Returns:
        DataFrame of the K nearest neighbors, including both immutable and non-immutable columns.
    """
    # Remove the immutable features from the data
    
    # Extract the relevant part of the query instance (excluding immutable columns)

    # print(query_instance)
    # query_instance = query_instance.to_frame().T
    # print(query_instance)
    non_immutable_columns = [col for col in data.columns if col not in immutable_columns]
    query_features = query_instance[non_immutable_columns]

    distances, indices = knn_tree.kneighbors(query_features.values, k)
    nearest_neighbors = pd.DataFrame(data.loc[indices[0]],columns= non_immutable_columns)
    for column in immutable_columns:
        nearest_neighbors[column] = query_instance[column].values[0]
    return indices, distances


def test_knn_tree():
    # Create a small mock dataset
    data = pd.DataFrame({
        'age': [25, 30, 22, 40, 35],    # immutable feature
        'income': [50000, 60000, 45000, 80000, 75000],
        'education': [12, 16, 14, 18, 16]
    })
    
    labels = pd.Series([0, 1, 1, 0, 1])  # Corresponding labels

    # Define immutable columns
    immutable_columns = ['age']
    
    # Step 1: Build the kNN tree for instances where labels == 1
    knn_tree, subset_data = build_knn_tree(
        data=data, 
        labels=labels, 
        counterfactual_class=1, 
        immutable_columns=immutable_columns,
        n_neighbors=3  # Set to 3 neighbors for testing
    )

    # Step 2: Create a query instance (it should match the structure of the data)
    query_instance = pd.Series({
        'age': 30,         # immutable feature
        'income': 55000,   # query feature
        'education': 14    # query feature
    })

    # Step 3: Find the 3 nearest neighbors excluding the immutable feature ('age')
    nearest_neighbors, distances = find_k_nearest_neighbors(
        data=subset_data, 
        knn_tree=knn_tree, 
        query_instance=query_instance, 
        k=3, 
        immutable_columns=immutable_columns
    )

    # Step 4: Print results for inspection
    print("Nearest Neighbors (including immutable features):")
    print(nearest_neighbors)
    
    print("\nDistances:")
    print(distances)

# This block ensures the test only runs when this script is executed directly.
if __name__ == "__main__":
    test_knn_tree()
