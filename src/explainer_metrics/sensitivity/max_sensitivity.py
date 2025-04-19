from quantus.metrics import MaxSensitivity
from src.explainer_metrics.explainer_metric import ExplainerMetric


class MaxSensitivityMetric(ExplainerMetric):
    """
    This class is a wrapper around the Complexity class from the quantus.metrics module.
    It is used to calculate nsthe complexity of a model's predictions.
    """

    def __init__(self, nr_samples):
        self.nr_samples = nr_samples
        super().__init__()

    def init_explainer_metric(
        self,
        model,
        data_batch,
        explanations,
        explainer_func, 
    ):
        """
        Initialize the complexity metric.

        Args:
            model: The model to evaluate.
            data_batch: The data batch to use for evaluation.
            explanations: The explanations to use for evaluation.
        """
        X, y = data_batch
        X, y = X.numpy(), y.numpy()
        self._calculate_metric(model, data_batch, explanations, explainer_func)

    def _calculate_metric(
            self,
            model,
            data_batch,
            explanations,
            explainer_func,
    ):
        """
        Calculate the complexity of the model's predictions.

        Args:
            model: The model to evaluate.
            data_batch: The data batch to use for evaluation.
            explanations: The explanations to use for evaluation.

        Returns:
            The complexity of the model's predictions.
        """
        X, y = data_batch
        X, y = X.numpy(), y.numpy()
        print(f"X shape: {X.shape}, y shape: {y.shape}, explanations shape: {explanations.shape}")
        self._calculated_metrics = MaxSensitivity(nr_samples=self.nr_samples)(
            model=model,
            x_batch=X[:len(explanations), None],  # Add a new dimension at axis 1
            y_batch=y[:len(explanations)],  
            a_batch=explanations[:, None],  # Add a new dimension at axis 1
            explain_func=explainer_func,
        )
        return self._calculated_metrics