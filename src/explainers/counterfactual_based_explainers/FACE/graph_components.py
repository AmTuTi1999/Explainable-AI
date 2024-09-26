from typing import Callable
import networkx as nx
import numpy as np
from src.explainers.counterfactual_based_explainers.FACE import face_utis as utils

def create_path_to_counterfactual_class(
        model: Callable,
        x_batch: np.ndarray,
        input_vector: np.ndarray, 
        counterfactual_target_class: int,
        graph: nx.Graph,
        visited: list[int],
        distance_threshold: float, 
        kernel_function: Callable,
):
    """_summary_

    Args:
        model (Callable): _description_
        x_batch (np.ndarray): _description_
        input_vector (np.ndarray): _description_
        counterfactual_target_class (int): _description_
        graph (nx.Graph): _description_
        visited (list[int]): _description_
        distance_threshold (float): _description_
        kernel_function (Callable): _description_

    Returns:
        _type_: _description_
    """    
    i = 0
    x_i = input_vector
    graph.add_node(utils.get_index(x_batch, x_i))
    while i < len(x_batch):
        x_j = utils.find_closest(x_i, x_batch, visited)
        i += 1
        if np.linalg.norm(x_i - x_j) < distance_threshold:
            graph.add_node(utils.get_index(x_batch, x_j))
            graph.add_edge(utils.get_index(x_batch, x_i), utils.get_index(x_batch, x_j), density = kernel_function(x_i, x_j))
            if model.predict(x_j) != counterfactual_target_class:
                x_i = x_j
            else:
                break
    return x_j, graph


def create_recourse_graph(
        x_batch,
        y_batch,
        input_vector,
        graph: nx.Graph,
        counterfactual_target_class,
        number_of_paths,
        visited: list,
        kernel_function: Callable,
        max_iter: int,
        check_constraints: Callable,
):
    """_summary_

    Args:
        x_batch (_type_): _description_
        y_batch (_type_): _description_
        input_vector (_type_): _description_
        graph (nx.Graph): _description_
        counterfactual_target_class (_type_): _description_
        number_of_paths (_type_): _description_
        visited (list): _description_
        kernel_function (Callable): _description_
        max_iter (int): _description_
        check_constraints (Callable): _description_

    Returns:
        _type_: _description_
    """  
    visited.append(utils.get_index(x_batch, input_vector))
    Ict = 0
    epoch = 0
    others = utils.find_neighbours(input_vector, x_batch[np.flatnonzero(y_batch == counterfactual_target_class)], visited)
    visited = visited + [utils.get_index(x_batch, arr) for arr in others]
    path_tensor = utils.generate_paths(
        others, x_batch[np.flatnonzero(y_batch == counterfactual_target_class)], number_of_paths, visited
    )
    path = np.zeros((1, x_batch.shape[1]))

    for i in range(path_tensor.shape[0]):
        graph.add_node(utils.get_index(x_batch, path_tensor[i][0]))
        graph.add_edge(utils.get_index(x_batch, input_vector), 
                       utils.get_index(x_batch, path_tensor[i][0]),
                       density = kernel_function(input_vector, path_tensor[i][0]) 
        )
        path_a = np.vstack([input_vector, path_tensor[i][0]])
        for k in range(1, path_tensor.shape[1]):
            
            if check_constraints(path_tensor[i][0], path_tensor[i][k], counterfactual_target_class):
                graph.add_node(utils.get_index(x_batch, path_tensor[i][k]))
                graph.add_edge(utils.get_index(x_batch, path_tensor[i][0]), 
                               utils.get_index(x_batch, path_tensor[i][k]),
                               density = kernel_function(path_tensor[i][0], path_tensor[i][k]) 
                )
                path = np.vstack([path_a,  path_tensor[i][k]])
                while epoch < max_iter:
                    new_others = utils.find_neighbours(
                        path[-1], x_batch[np.flatnonzero(y_batch == counterfactual_target_class)], visited
                    )
                    visited = visited + [utils.get_index(x_batch, arr) for arr in new_others]
                    path_proxy = []
                    
                    for i, n_i in enumerate(new_others):
                        if check_constraints(path[-1], n_i, counterfactual_target_class):
                            Ict += 1
                            if np.array([np.array_equal(n_i, p) for p in path]).sum() == 0:
                                if utils.get_index(x_batch, path[-1]) != utils.get_index(x_batch,path_tensor[i][k]):
                                    graph.add_node(utils.get_index(x_batch, path[-1]))
                                graph.add_node(utils.get_index(x_batch, n_i))
                                graph.add_edge(utils.get_index(x_batch, path[-1]),
                                               utils.get_index(x_batch, n_i), 
                                               density = kernel_function(path_tensor[i][k], path[-1])
                                )
                                path = np.vstack([path,  n_i])
                                if epoch == max_iter - 1:
                                    path_proxy.append(path)
                                visited.append(utils.get_index(x_batch,path[-1]))
                        else:
                            break       
                    epoch += 1

    counterfactuals_indices = [node for node in graph.nodes if graph.out_degree(node) == 0]

    return counterfactuals_indices, graph