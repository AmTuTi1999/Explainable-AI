import logging
import networkx as nx
import numpy as np
from src.explainers.helpers.helpers import get_opposite_class
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.counterfactual_based_explainers.FACE import kernel
from src.explainers.counterfactual_based_explainers.FACE.graph_components import create_path_to_counterfactual_class, create_recourse_graph

class FACE(CounterfactualExplainerBase):
    """_summary_

    Args:
        CounterfactualExplainerBase (_type_): _description_
    """    
    def __init__(
            self, 
            model, 
            x_batch,
            y_batch,
            epsilon= 0.1, 
            density_threshold = 0.7,  
            number_of_paths: int= 5,
            categorical_features: list[str] = None,
            feature_names: dict= None,
            discretize_continuous: bool = False,
            discretizer: str = 'decile', 
            classification_threshold: float = 0.5, 
            immutable_features: list[str]= None, 
            density_estimator: str = 'kde', 
            kde_bandwidth: float = 0.1, 
            knn_number_of_points: int = 5, 
            knnK: int = 5, 
            knn_volume: float = 0.3
    ):
        super().__init__(
            model= model,
            x_batch= x_batch,
            y_batch= y_batch,
            immutable_features= immutable_features,
            categorical_features = categorical_features,
            feature_names = feature_names,
            discretize_continuous=discretize_continuous,
            discretizer=discretizer
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
        """_summary_

        Args:
            input_vector (_type_): _description_
            counterfactual_target_class (_type_): _description_

        Returns:
            _type_: _description_
        """        
        graph = nx.DiGraph()
        instance_class = self.model.predict(input_vector.to_frame().T)
        input_vector = self.explainer_first_step(input_vector)
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
        return counterfactuals, graph
