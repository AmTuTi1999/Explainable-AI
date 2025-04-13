class check_sparsity:


    def __init__(self, max_number_of_features):
        self.m = max_number_of_features

    

    def l0_norm(self, vector):
        """
        Calculate the L0 norm of a vector, which is the count of non-zero elements in the vector.

        Args:
        - vector: The input vector (list or array).

        Returns:
        - The L0 norm of the vector (count of non-zero elements).
        """
        count = 0
        for element in vector:
            if element != 0:
                count += 1
        return count
    

    def is_sparse(self,unit,  counterfactual):
        diff = unit - counterfactual
        if self.l0_norm(diff) < self.m:
            return True
        else:
            return False
        



class CustomScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, arr):
        # Find the minimum and maximum values in the array
        self.min_val = min(arr)
        self.max_val = max(arr)

    def transform(self, data_point):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform.")

        # Check if the range is zero to avoid division by zero
        if self.min_val == self.max_val:
            return 0.5  # Return 0.5 for a single point when range is zero, placing it in the middle of [0, 1]

        # Scale the data point to the range [0, 1]
        scaled_value = (data_point - self.min_val) / (self.max_val - self.min_val)
        
        return scaled_value
    


def generate_subsets(nums):
    def backtrack(start, current_subset):
        # Add the current subset to the list of subsets
        subsets.append(current_subset[:])
        
        # Explore all possible options to form subsets
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()

    subsets = []
    backtrack(0, [])
    return subsets


def weighted_l1_norm(x1, x2):
    s = 0
    n = len(x2)
    for i in range(n):
        s += abs(x1[i] - x2[i])
    return s/n

import pandas as pd

def extend_dataframe(df1, df2):
    """
    Extends the first DataFrame with rows from the second DataFrame.
    Both DataFrames must have the same columns.
    
    Parameters:
    - df1: The original DataFrame to extend.
    - df2: The DataFrame to append to df1.
    
    Returns:
    - A new DataFrame that is the result of concatenating df1 and df2.
    """
    # Ensure that both DataFrames have the same columns
    if list(df1.columns) != list(df2.columns):
        raise ValueError("DataFrames must have the same columns to concatenate.")
    
    # Concatenate the two DataFrames
    extended_df = pd.concat([df1, df2], ignore_index=True)
    
    return extended_df

def get_opposite_class(binary_class):
    """
    Returns the opposite class for a binary classification input (0 or 1).
    
    Parameters:
    - binary_class: The input binary class (0 or 1).
    
    Returns:
    - The opposite binary class (1 if input is 0, 0 if input is 1).
    """
    if binary_class not in [0, 1]:
        raise ValueError("Input must be either 0 or 1 for binary classification.")
    
    return 1 - binary_class

