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
        print(counterfactuals_indices)
        return counterfactuals, graph
