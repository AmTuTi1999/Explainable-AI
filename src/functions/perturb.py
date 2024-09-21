import random 
import numpy as np
from src.explainers.helpers.assertions import assert_2d

seed = 42
rng = np.random.default_rng(seed)

import itertools
import random

def generate_random_permutations(input_list):
    result = []
    # Iterate over lengths from 1 to n
    for r in range(0, len(input_list)):
        # Get all permutations of length r and convert them to lists
        permutations = [list(perm) for perm in itertools.permutations(input_list, r)]
        result.extend(permutations)
    
    # Shuffle the entire result to make it random
    random.shuffle(result)
    
    return result


def gaussian_2d(
        x_batch,
        mean: int = 0, 
        std_dev: int = 1,
    ):
    """
    Perturb a vector by replacing a specified number of its values with random samples
    from a Gaussian distribution.

    Args:
    - vector: The input vector (list) to be perturbed.
    - num_to_perturb: The number of values to perturb.
    - mean: The mean of the Gaussian distribution (default is 0).
    - std_dev: The standard deviation of the Gaussian distribution (default is 1).

    Returns:
    - The perturbed vector with some values replaced by Gaussian samples.
    """

    assert_2d(x_batch)
    x_batch = np.array(x_batch)
    _ , num_features = x_batch.shape
    
    # Ensure the input is a NumPy array
    x_batch = np.array(x_batch)

    perturbed_batch = x_batch.copy()
    
    # Generate Gaussian noise with the same shape as the input vectors
    noise = np.random.normal(loc=mean, scale=std_dev, size=x_batch.shape)
    
    # Choose random indices to perturb
    columns_to_perturb = generate_random_permutations(range(num_features))

    # Perturb the selected indices
    for columns in columns_to_perturb:
        perturbed_batch[:, columns] = perturbed_batch[:, columns].astype(float) + noise[:, columns]
    return perturbed_batch



def towards_class_2d(
        x_batch,
        y_batch,
        target_class: int,
    ):
    
    assert_2d(x_batch)
    x_batch = np.array(x_batch)
    _ , num_features = x_batch.shape
    
    # Create a copy of the input vector to avoid modifying the original
    perturbed_batch = x_batch.copy()

    counterfactual_class = x_batch[y_batch == target_class]
    mean_counterfactual_class= counterfactual_class.mean(axis = 0)
    perturbation_factor = rng.uniform(low = 0, high = 1)
    difference_vector = -1*perturbed_batch + mean_counterfactual_class

    # Choose random indices to perturb
    columns_to_perturb = generate_random_permutations(range(num_features))

    # Perturb the selected indices
    for columns in columns_to_perturb:
        
        perturbed_batch[:, columns] = perturbed_batch[:, columns] + perturbation_factor * difference_vector[:, columns]
    return perturbed_batch


def categorical_2d(x_batch):
    """
    Perturb a batch of categorical NumPy arrays by randomly selecting new values from
    the set of unique categories in each column.
    
    Parameters:
    batch (np.ndarray): A 2D NumPy array where each row is a sample and each column is a category.
    perturbation_prob (float): The probability of perturbing each element (default is 0.1).
    
    Returns:
    np.ndarray: A perturbed version of the input categorical batch.
    """
    # Ensure it's a NumPy array
    assert_2d(x_batch)
    x_batch = np.array(x_batch)
    _ , num_features = x_batch.shape

    # Copy the batch to avoid modifying the original
    perturbed_batch = x_batch.copy()
    
    # Choose random indices to perturb
    columns_to_perturb = generate_random_permutations(range(num_features))
    # Iterate through each column (feature)
    for col in columns_to_perturb:
        # Get the unique categories in the column
        unique_categories = np.unique(perturbed_batch[:, col])
        perturbed_batch[:, col] = random.choices(unique_categories, k=len(perturbed_batch))
    return perturbed_batch

