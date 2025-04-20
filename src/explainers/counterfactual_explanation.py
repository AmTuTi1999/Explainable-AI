from numpy.typing import NDArray
import networkx as nx
from typing import Union
class CounterfactualExplanation:
    """
    Class for generating counterfactual explanations.
    """

    def __init__(
            self, 
            input_vector: NDArray, 
            counterfactuals: NDArray,
            feature_names: list[str],
            actual_class: Union[int, str],
            counterfactual_target_class: Union[int, str],
            graph: nx.DiGraph = None,
            counterfactual_predictions = None,
        ) -> None:
        """
        Initialize the CounterfactualExplanation class.

        Parameters:
        - model: The model to be explained.
        - data: The data used for generating explanations.
        """
        self.input_vector = input_vector
        self.counterfactuals = counterfactuals[:, None] if counterfactuals.ndim == 2 else counterfactuals.reshape(1, -1)[:, None]
        self.feature_names = feature_names
        self.actual_class = actual_class
        self.counterfactual_target_class = counterfactual_target_class
        self._graph = graph
        self.counterfactual_predictions = counterfactual_predictions



    def _get_changed_columns(self):
        mask = self.input_vector != self.counterfactuals
        columns = [feature for feature, changed in zip(self.feature_names, mask) if changed]
        return columns