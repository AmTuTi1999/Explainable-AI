from quantus.metrics import Complexity

from src.explainer_metrics.explainer_metric import ExplainerMetric

class ComplexityMetric(ExplainerMetric):
    """
    This class is a wrapper around the Complexity class from the quantus.metrics module.
    It is used to calculate nsthe complexity of a model's predictions.
    """

    def __init__(self, normalise):
        self.normalise = normalise
        super().__init__()

    def init_explainer_metric(
        self,
        model,
        data_batch,
        explanations,
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
        self._calculate_metric(model, data_batch, explanations)

    def _calculate_metric(
            self,
            model,
            data_batch,
            explanations,
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
        self._calculated_metrics = Complexity(normalise=True)(
            model=model,
            x_batch=X,
            y_batch=y,  
            a_batch=explanations,
        )
        return self._calculated_metrics