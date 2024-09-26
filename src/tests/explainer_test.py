import unittest
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression  
import pandas as pd
import numpy as np
import networkx as nx
from src.explainers.counterfactual_based_explainers.MACE.mace import MACE
from src.explainers.counterfactual_based_explainers.FACE.face import FACE
from src.explainers.counterfactual_based_explainers.CERTIFAI.certifai import CERTIFAI




# Define a mock classifier model for testing
class MockModel:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def predict(self, x):
        """
        Mock predict function that simply classifies based on the sum of the input vector.
        """
        if isinstance(x, pd.DataFrame):
            # Sum over each row and compare to threshold
            return np.where(x.sum(axis=1) > self.threshold, 1, 0)
        else:
            # Handle single instance (1D array)
            return np.array([1 if np.sum(x) > self.threshold else 0])

# Define a mock distance function
def mock_distance_function(x1, x2):
    return np.linalg.norm(np.array(x1) - np.array(x2))

# Mock implementation of CounterfactualExplainerBase to avoid inheritance issues
class CounterfactualExplainerBase:
    def __init__(self, model, x_batch, y_batch):
        self.model = model
        self.x_batch = x_batch
        self.y_batch = y_batch

    def explainer_first_step(self, input_vector):
        """
        Mock explainer_first_step function to simulate transformation of the input_vector.
        """
        return input_vector, None

def get_opposite_class(instance_class):
    return 1 if instance_class == 0 else 0

# Import the CERTIFAI class (assuming it was defined as shown in the earlier implementation)
# from certifai_module import CERTIFAI  # Uncomment this if CERTIFAI is in a separate module

class TestCERTIFAI(unittest.TestCase):

    def setUp(self):
        """
        Set up mock data for testing CERTIFAI.
        """
        self.model = MockModel(threshold=10)
        
        # Simulating a batch of input vectors (x_batch) and their respective classes (y_batch)
        self.x_batch = pd.DataFrame(np.random.uniform(0, 10, (100, 5)))
        self.y_batch = pd.Series([1 if np.sum(x) > 10 else 0 for x in self.x_batch.values])
        
        # Search space for each feature (min and max values)
        
        self.certifai = CERTIFAI(
            model=self.model,
            x_batch=self.x_batch,
            y_batch=self.y_batch,
            distance_function=mock_distance_function,
            mutation_rate=0.1,
            crossover_rate=0.5,
            generations=10,
            population_size=5
        )

    def test_explain_instance_binary_class(self):
        """
        Test the explain_instance function for binary classification.
        """
        # Use a sample input vector
        input_vector = pd.Series(np.random.uniform(0, 10, 5))

        # Test the explain_instance function with counterfactual target class
        counterfactual_target_class = 'opposite'

        # Patch the logging to avoid actual log output during test
        with unittest.mock.patch('logging.info'):
            counterfactual = self.certifai.explain_instance(input_vector, counterfactual_target_class)

        # Ensure the output is not None and has the same dimensionality
        self.assertIsNotNone(counterfactual)
        self.assertEqual(len(counterfactual), len(input_vector))
        print(counterfactual)
        # Check if the counterfactual class is different from the original class
        original_class = self.model.predict(input_vector.to_frame().T)
        counterfactual_class = self.model.predict(counterfactual.reshape(1, -1))

        self.assertNotEqual(original_class, counterfactual_class)

    def test_fitness_function(self):
        """
        Test the _fitness function to ensure it calculates the distance correctly.
        """
        input_vector = np.random.uniform(0, 10, 5)
        candidate = np.random.uniform(0, 10, 5)

        fitness_score = self.certifai._fitness(input_vector, candidate)

        # Ensure the fitness score is a positive value
        self.assertGreater(fitness_score, 0)

if __name__ == '__main__':
    unittest.main()



class TestFACE(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and sample data for testing."""
        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.predict_proba = MagicMock(return_value=[0.2, 0.8])  # Mock probability
        self.mock_model.predict = MagicMock(return_value=[1])  # Mock class prediction

        # Mock kernel function
        self.mock_kernel = MagicMock()
        self.mock_kernel_function = MagicMock(return_value=0.6)  # Mock some density value
        
        # Sample data
        self.x_batch = pd.DataFrame(np.random.rand(5, 3))  # Random input batch
        self.y_batch = pd.Series([0, 1, 0, 1, 0])  # Random output batch
        
        
        # Initialize FACE instance
        self.face = FACE(
            model=self.mock_model,
            x_batch=self.x_batch,
            y_batch=self.y_batch,
            epsilon=0.1,
            immutable_features=None,
            density_threshold=0.7,
            number_of_paths=3,
            classification_threshold=0.5,
            density_estimator="kde",
            kde_bandwidth=0.1
        )
        # Set kernel function for testing
        self.face.kernel_function = self.mock_kernel_function

    def test_check_constraints(self):
        """Test the check_constraints method with mocked values."""
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.15, 0.25, 0.35])
        counterfactual_target_class = 1
        
        # Call check_constraints
        result = self.face.check_constraints(a, b, counterfactual_target_class)
        
        # Assertions
        self.mock_kernel_function.assert_called_once_with(a, b)
        self.mock_model.predict_proba.assert_called_once_with(b)
        
        # Since the distance is small but classified probability is above the threshold,
        # the result should be None (not meeting constraints)
        self.assertIsNone(result)

    def test_explain_instance(self):
        """Test explain_instance method and ensure graph and counterfactuals are generated."""
        input_vector = pd.Series([0.1, 0.2, 0.3])
        counterfactual_target_class = 'opposite'
        
        # Mock helper methods
        self.face.explainer_first_step = MagicMock(return_value=(input_vector, None))
        self.face.model.predict = MagicMock(return_value=[0])  

        # Call explain_instance
        counterfactuals, graph = self.face.explain_instance(input_vector, counterfactual_target_class)
        
        # Assertions
        self.face.explainer_first_step.assert_called_once_with(input_vector)
        #self.face.model.predict.assert_called_once()
        
        self.assertIsInstance(counterfactuals, pd.DataFrame)
        self.assertIsInstance(graph, nx.DiGraph)  # We expect two counterfactuals returned


def test_mace():
    # Sample Data
    x_batch = pd.DataFrame({
        'age': [25, 30, 22, 40, 35],
        'income': [50000, 60000, 45000, 80000, 75000],
        'education': [12, 16, 14, 18, 16]
    })
    y_batch = pd.Series([0, 1, 1, 0, 1])

    # Initialize mock model and MACE instance
    model = LogisticRegression()
    model.fit(x_batch[['age', 'income', 'education']], y_batch)
    mace = MACE(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        top_num_features=2,
        top_num_feature_values=3,
        num_points_neighbourhood=3,
        immutable_features=['age'],
        gamma=0.1,
        alpha=0.1,
        num_episodes=100,
        lambdas=(0.1, 0.1),
        sparsity_constraint=15,
        num_counterfactuals=5,
        max_search_radius=0.1,
        min_search_radius=0.01,
        refine_epochs=10
    )

    # Test explain_instance method
    input_vector = pd.Series({'age': 30, 'income': 55000, 'education': 14})
    counterfactuals = mace.explain_instance(
        input_vector=input_vector,   # For this test, the neighbor_data is not used
        counterfactual_target_class='opposite'
    )

    # Print results
    print("Counterfactuals:")
    print(counterfactuals)

    # Validate outputs (add more validations as needed)
    assert isinstance(counterfactuals, pd.DataFrame), "Counterfactuals should be a DataFrame"
    assert len(counterfactuals) == mace.b, f"Expected {mace.b} counterfactuals"

if __name__ == "__main__":
    test_mace()
    unittest.main()
    print('DONE')