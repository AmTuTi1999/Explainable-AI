import numpy as np   

def get_index(data, target_array):
    for i, row in enumerate(data):
        if all(target_array == row):
            return i  # Return the index of the row if the array is found
    return -1 

def find_closest(unit, data, already_visited: list):
    index = get_index(data, unit)
    already_visited.append(index)
    distances = [np.linalg.norm(unit - data[i]) for i in range(len(data)) if i not in already_visited]
    wanted_index = np.argsort(np.array(distances))[0]
    return data[wanted_index]

def find_neighbours(unit, data, already_visited: list):
    wanted_indices = [i for i in range(len(data)) if i not in already_visited]
    distances = [np.linalg.norm(unit - data[wanted_indices][i]) for i in range(len(data[wanted_indices]))]
    wanted_index = np.argsort(np.array(distances))
    return data[wanted_indices][wanted_index]

def generate_paths(node_set, data, number_of_paths, visited_nodes: list):
    net = np.ones((len(node_set), number_of_paths + 1, data.shape[1]))
    print(net.shape)
    print(node_set)
    for i, n_i in enumerate(node_set):
        neighbours = find_neighbours(n_i, data, visited_nodes)[1: number_of_paths]
        for blet in neighbours:
            generated_nodes = np.vstack([n_i, find_neighbours(blet, data, visited_nodes)])
        net[i] = generated_nodes
    return net
    

