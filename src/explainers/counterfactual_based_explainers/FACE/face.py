import networkx as nx
import numpy as np
import logging
import pandas as pd
from src.explainers.helpers.helpers import get_opposite_class
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.counterfactual_based_explainers.FACE import kernel
from src.explainers.counterfactual_based_explainers.FACE.graph_components import create_path_to_counterfactual_class, create_recourse_graph

class FACE(CounterfactualExplainerBase):
    def __init__(
            self, 
            model, 
            x_batch,
            y_batch,
            epsilon, 
            density_threshold,  
            number_of_paths, 
            classification_threshold = 0.5, 
            immutable_features= None, 
            density_estimator = None, 
            kde_bandwidth = None, 
            knn_number_of_points = None, 
            knnK = None, 
            knn_volume = None
    ):
        super().__init__(
            model= model,
            x_batch= x_batch,
            y_batch= y_batch,
            immutable_features= immutable_features,
            epsilon = epsilon, 
            density_threshold = density_threshold,  
            number_of_paths= number_of_paths, 
            classification_threshold= classification_threshold,  
            density_estimator= density_estimator, 
            kde_bandwidth= kde_bandwidth, 
            knn_number_of_points= knn_number_of_points, 
            knnK= knnK, 
            knn_volume= knn_volume
    )
        self.model = model
        self.t_d = density_threshold
        self.t_p = classification_threshold
        self.model = model
        self.n = number_of_paths
        self.eps = epsilon
        
        self.stored_indices = []
        self.visited = []
        if density_estimator == "kde":
            self.kernel_function = kernel.KDE(
                x_batch, 
                kde_bandwidth, 
                epsilon
            )
        if density_estimator == "knn":
            self.kernel_function = kernel.KNN(
                x_batch,
                knn_volume,
                knnK,
                knn_number_of_points
            )


    def check_constraints(
            self, 
            a, 
            b, 
            counterfactual_target_class
        
        ):
        """_summary_

        Args:
            a (_type_): _description_
            b (_type_): _description_
            counterfactual_target_class (_type_): _description_

        Returns:
            _type_: _description_
        """        
        density = self.kernel_function(a, b)
        dist = np.linalg.norm(a-b)
        classified_prob = self.model.predict_proba(b)[counterfactual_target_class]
        if dist< self.eps:
            if density > self.t_p:
                if classified_prob < self.t_d:
                    return True

    def explain_instance(
            self, 
            input_vector, 
            counterfactual_target_class
    ):
        graph = nx.DiGraph()
        instance_class = self.model.predict(input_vector.to_frame().T)
        input_vector, _ = self.explainer_first_step(input_vector)
        if counterfactual_target_class == 'opposite':
            logging.info("Calling Explainer for Binary Class")
            counterfactual_target_class = get_opposite_class(instance_class)
        x_j, graph = create_path_to_counterfactual_class(
            self.model, np.array(self.x_batch), input_vector, counterfactual_target_class, graph, self.visited, self.t_d, self.kernel_function
        )
        counterfactuals_indices, graph = create_recourse_graph(
            np.array(self.x_batch), np.array(self.y_batch), x_j, graph, counterfactual_target_class, self.n, self.visited, self.kernel_function, 10, self.check_constraints
        )

        counterfactuals = self.x_batch.iloc[counterfactuals_indices]
        print(counterfactuals_indices)
        return counterfactuals, graph
    


import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import networkx as nx

# Assuming the FACE class and other depe

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
        # Mock the methods used within the explain_instance
        create_path_to_counterfactual_class = MagicMock(return_value=(input_vector, nx.DiGraph()))
        create_recourse_graph = MagicMock(return_value=([0, 2], nx.DiGraph()))
        
        # Call explain_instance
        counterfactuals, graph = self.face.explain_instance(input_vector, counterfactual_target_class)
        
        # Assertions
        self.face.explainer_first_step.assert_called_once_with(input_vector)
        self.face.model.predict.assert_called_once()
        
        self.assertIsInstance(counterfactuals, pd.DataFrame)
        self.assertIsInstance(graph, nx.DiGraph)  # We expect two counterfactuals returned


if __name__ == '__main__':
    unittest.main()
    print('DONE')
