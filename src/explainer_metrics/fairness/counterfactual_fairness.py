from quantus.metrics.base import Metric
from quantus.helpers.enums import (
    DataType,
    ModelType,
    ScoreDirection,
)
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

class CounterfactualFairness(Metric):
    """
    Counterfactual Fairness class.
    """
    name = "Counterfactual Fairness"
    data_applicability = {DataType.TIMESERIES, DataType.TABULAR}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.LOWER
    evaluation_category = "Fairness"

    def __init__(
        self, 
        sensitive_features: List[str],
        feature_names: List[str],
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
        ):
        """
        Initialize the CounterfactualFairness class.

        Parameters
        ----------
        model : object
            The model to be evaluated.
        sensitive_features : list
            The sensitive features to be evaluated.
        """
        self.senisitive_features = sensitive_features
        self.feature_names = feature_names
        super().__init__(
            abs=False,
            normalise=False,
            normalise_func=None,
            normalise_func_kwargs=None,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray = None,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        self.model = model
        return super().__call__(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    s_batch=s_batch,
                    custom_batch=None,
                    channel_first=channel_first,
                    explain_func=explain_func,
                    explain_func_kwargs=explain_func_kwargs,
                    softmax=softmax,
                    device=device,
                    model_predict_kwargs=model_predict_kwargs,
                    batch_size=batch_size,
                **kwargs,
            )

    def evaluate_instance(self, x: np.ndarray, a: np.ndarray) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.

        Returns
        -------
        float
            The evaluation results.
        """

        if len(x.shape) == 1:
            newshape = np.prod(x.shape)
        else:
            newshape = np.prod(x.shape[1:])
        original_prediction = self.model.predict(torch.tensor(a, dtype=torch.float32)) 
        attribution_df = self.convert_to_df(a.reshape(1, -1))
        # Permute the feature values of sensitive features.
        fair_predictions = []
        for feature in self.senisitive_features:
            unique_values = np.unique(attribution_df[feature].values)
            for value in unique_values:
                temp_df = attribution_df.copy()
                temp_df[feature] = value
                permuted_prediction = self.model.predict(torch.tensor(temp_df.to_numpy(), dtype=torch.float32))
                fair_predictions.append(permuted_prediction == original_prediction)
        return np.array(fair_predictions).mean()

    def evaluate_batch(
        self, x_batch: np.ndarray, a_batch: np.ndarray, **kwargs
    ) -> List[float]:
        """
        This method performs XAI evaluation on a single batch of explanations.
        For more information on the specific logic, we refer the metric’s initialisation docstring.

        Parameters
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        kwargs:
            Unused.

        Returns
        -------
        scores_batch:
             The evaluation results.
        """
        return [self.evaluate_instance(x=x, a=a) for x, a in zip(x_batch, a_batch)]

    
    def convert_to_df(self, x):
        df = pd.DataFrame(x, columns=self.feature_names)
        return df
    