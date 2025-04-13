import unittest
import numpy as np
import pandas as pd
from src.explainer_metrics.fairness.counterfactual_fairness import CounterfactualFairness
import torch

import torch.nn as nn

class MockModel(nn.Module):
    """
    A mock PyTorch model for testing purposes.
    """
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = nn.Linear(5, 1)  # Assuming input has 5 features

    def predict(self, x):
        """
        Forward pass that applies a simple linear transformation and threshold.
        """
        x = self.linear(x)
        return (x > 0).float()  # Return 1 if output > 0, else 0

class TestCounterfactualFairness(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment with a mock model and sample data.
        """
        self.mock_model = MockModel()
        self.sensitive_features = ['gender', 'race']
        self.feature_names = ['age', 'income', 'education', 'gender', 'race']
        self.metric = CounterfactualFairness(
            sensitive_features=self.sensitive_features,
            feature_names=self.feature_names
        )

    # def test_evaluate_instance(self):
    #     """
    #     Test the evaluate_instance method.
    #     """
    #     # Sample input and explanation
    #     x = np.array([25, 50000, 16, 0, 1])
    #     a = np.array([25, 50000, 16, 0, 1])

    #     # Call evaluate_instance
    #     fairness_score = self.metric.evaluate_instance(x=x, a=a)

    #     # Assert the fairness score is between 0 and 1
    #     self.assertGreaterEqual(fairness_score, 0)
    #     self.assertLessEqual(fairness_score, 1)

    def test_evaluate_batch(self):
        """
        Test the evaluate_batch method.
        """
        # Sample batch input and explanations
        x_batch = torch.tensor(np.array([
            [25, 50000, 16, 0, 1],
            [30, 60000, 18, 1, 0],
            [22, 45000, 14, 0, 1]
        ]))
        a_batch = np.array([
            [25, 50000, 16, 0, 1],
            [30, 60000, 18, 1, 0],
            [22, 45000, 14, 0, 1]
        ])

        # Call evaluate_batch
        fairness_scores = self.metric(model=self.mock_model, x_batch=x_batch, a_batch=a_batch)

        # Assert the fairness scores are between 0 and 1
        for score in fairness_scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_convert_to_df(self):
        """
        Test the convert_to_df method.
        """
        # Sample input
        x = np.array([
            [25, 50000, 16, 0, 1],
            [30, 60000, 18, 1, 0]
        ])

        # Call convert_to_df
        df = self.metric.convert_to_df(x)

        # Assert the DataFrame has the correct columns and values
        self.assertListEqual(df.columns.tolist(), self.feature_names)
        self.assertEqual(df.iloc[0]['age'], 25)
        self.assertEqual(df.iloc[1]['income'], 60000)

if __name__ == '__main__':
    unittest.main()