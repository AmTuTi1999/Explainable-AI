import logging
import numpy as np
from src.explainers.helpers.helpers import get_opposite_class
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.counterfactual_based_explainers.CERTIFAI.certifai_utils import GeneticAlgorithm

class CERTIFAI(CounterfactualExplainerBase):
    """_summary_

    Args:
        CounterfactualExplainerBase (_type_): _description_
    """    
    def __init__(
            self, 
            model, 
            x_batch,
            y_batch,
            distance_function, 
            mutation_rate=0.1, 
            crossover_rate=0.5, 
            generations=100, 
            population_size=50
    ):
        """
        Initialize the genetic algorithm.

        :param classifier: Black-box classifier function f
        :param x: Input instance for which we want to generate counterfactuals
        :param distance_function: A function that calculates the distance between two points
        :param search_space: Predefined search space W for all features (min and max values for each feature)
        :param mutation_rate: Probability of mutation (pm)
        :param crossover_rate: Probability of crossover (pc)
        :param generations: Number of generations to evolve
        :param population_size: Number of individuals in the population
        """
        super().__init__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch 
        )
        self.model = model
        self.distance_function = distance_function# predefined space for each feature
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population_size = population_size


    def _fitness(self, input_vector, candidate):
        """
        Compute the fitness of an individual as the inverse of the distance to the original input x.

        :param c: A candidate counterfactual
        :return: Fitness score (higher is better)
        """
        distance = self.distance_function(input_vector, candidate)
        return 1 / (distance + 1e-6)  # Adding small value to prevent division by zero

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
        instance_class = self.model.predict(input_vector.to_frame().T)
        input_vector = self.explainer_first_step(input_vector)
        if counterfactual_target_class == 'opposite':
            logging.info("Calling Explainer for Binary Class")
            counterfactual_target_class = get_opposite_class(instance_class)
        search_space = self.x_batch.to_numpy()[np.flatnonzero(self.y_batch.to_numpy() == counterfactual_target_class)]
        generator = GeneticAlgorithm(
            classifier=self.model.predict, 
            input_vector=input_vector,
            population_size=self.population_size,
            generations=self.generations,
            fitness_function=self._fitness,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            search_space=search_space,
        )
        counterfactual = generator.evolve()
        return counterfactual

