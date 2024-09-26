import numpy as np
import random

class GeneticAlgorithm:
    def __init__(
        self, 
        classifier, 
        input_vector, 
        population_size, 
        generations, 
        fitness_function, 
        mutation_rate, 
        crossover_rate, 
        search_space
    ):
        """
        Initialize the genetic algorithm.
        """
        self.classifier = classifier
        self.input_vector = input_vector
        self.population = search_space  # Initial population
        self.population_size = population_size
        self.generations = generations
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.search_space = search_space

    def select(self):
        """
        Select individual indices based on fitness probabilities.
        :return: Index of the selected individual
        """
        # Compute fitness for each individual in the population
        fitness_scores = np.array([self.fitness_function(self.input_vector, ind) for ind in self.population])
        probabilities = fitness_scores / np.sum(fitness_scores)
        return np.random.choice(range(len(self.population)), p=probabilities)

    def mutate(self, individual):
        """
        Mutate an individual by randomly adjusting its feature values within the predefined search space.
        :param individual: The individual to mutate
        :return: Mutated individual
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                low, high = self.search_space[i].min(), self.search_space[i].max()  # Bounds for the feature
                individual[i] = np.clip(individual[i] + np.random.uniform(-0.1, 0.1), low, high)
        return individual

    def crossover(self, parents_indices):
        """
        Perform crossover between multiple individuals. Each feature of the child is randomly chosen from one of the parents.
        :param parents_indices: List of parent indices from the population
        :return: A child individual created through random feature crossover from multiple parents
        """
        # Retrieve the actual parents based on their indices
        parents = np.array(self.population)[parents_indices]
        num_features = parents.shape[1]
        child = np.zeros(num_features)

        # Randomly choose each feature from one of the parents
        for i in range(num_features):
            selected_parent = random.choice(parents)
            child[i] = selected_parent[i]

        return child

    def evolve(self):
        """
        Run the genetic algorithm to evolve the population and find counterfactuals.
        :return: The best counterfactual found
        """
        for _ in range(self.generations):
            new_population = []

            while len(new_population) < self.population_size:
                # Select multiple parents for crossover by their indices
                num_parents = random.randint(2, 4)  # Choose between 2 to 4 parents randomly
                parents_indices = [self.select().item() for _ in range(num_parents)]

                # Perform crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parents_indices)
                else:
                    # If crossover doesn't happen, just select the first parent by index
                    child = self.population[parents_indices[0]].copy()

                # Mutate the child
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                # Ensure valid individuals (that satisfy the counterfactual constraint)
                if self.classifier(child) != self.classifier(self.input_vector):
                    new_population.append(child)

            # Update population
            self.population = new_population

            # Early stopping if the population is empty (rare)
            if len(self.population) == 0:
                break

        # Return the best individual based on fitness score
        best_individual = max(self.population, key=lambda c: self.fitness_function(self.input_vector, c))
        return best_individual


def initialize_search_space(
        x_batch: np.ndarray, 
        y_batch: np.ndarray, 
        counterfactual_target_class: int
    ):
    """
    Initialize search space based on the target class.
    :param x_batch: Batch of input features
    :param y_batch: Corresponding output labels
    :param counterfactual_target_class: The target class for the counterfactual
    :return: The subset of x_batch belonging to the counterfactual target class
    """
    search_space = x_batch[np.flatnonzero(y_batch == counterfactual_target_class)]
    return search_space
