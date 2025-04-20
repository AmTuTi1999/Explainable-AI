import logging
import numpy as np
from src.explainers.helpers.helpers import get_opposite_class
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.counterfactual_based_explainers.CERTIFAI.certifai_utils import GeneticAlgorithm
from src.explainers.counterfactual_explanation import CounterfactualExplanation

class CERTIFAI(CounterfactualExplainerBase):
    """_summary_

    Args:
        CounterfactualExplainerBase (_type_): _description_
    """    
    def __init__(
            self, 
            distance_function, 
            discretize_continuous = False,
            discretizer = 'decile',
            mutation_rate=0.1, 
            crossover_rate=0.5, 
            generations=100, 
            population_size=50,
            feature_names = None,
            categorical_features: list[str] = None, 
            immutable_features: list[str]= None, 
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
            immutable_features=immutable_features,
            categorical_features = categorical_features,
            feature_names = feature_names,
            discretize_continuous=discretize_continuous,
            discretizer=discretizer
        )
        if isinstance(distance_function, str):
            if distance_function == 'l1':
                self.distance_function = lambda x, y: np.sum(np.abs(x - y))
            elif distance_function == 'l2':
                self.distance_function = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
            elif distance_function == 'linf':
                self.distance_function = lambda x, y: np.max(np.abs(x - y))
            else:
                raise ValueError(f"Unsupported distance function: {distance_function}")
        else:
            self.distance_function = distance_function  # Assume it's a callable
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population_size = population_size


    def init_explainer(
            self,
            model,
            x_batch,                
            y_batch,
            x_batch_stats = None,
    ):
        self._init_explainer(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            x_batch_stats=x_batch_stats,
        )
        

    def _fitness(self, input_vector, candidate):
        """
        Compute the fitness of an individual as the inverse of the distance to the original input x.

        :param c: A candidate counterfactual
        :return: Fitness score (higher is better)
        """
        distance = self.distance_function(input_vector, candidate)
        return 1 / (distance + 1e-6)  # Adding small value to prevent division by zero

    def alias(self):
        return "CERTIFAI"
    
    def explain_instance(
            self, 
            input_vector, 
            counterfactual_target_class
    ) -> CounterfactualExplanation: 
        """_summary_

        Args:
            input_vector (_type_): _description_
            counterfactual_target_class (_type_): _description_

        Returns:
            _type_: _description_
        """        
        instance_class = self.model.predict(input_vector.to_frame().T)
        input_vector_rev = self.explainer_first_step(input_vector)
        if counterfactual_target_class == 'opposite':
            counterfactual_target_class = get_opposite_class(instance_class)
        search_space = self.x_batch.to_numpy()[np.flatnonzero(self.y_batch.to_numpy() == counterfactual_target_class)]
        generator = GeneticAlgorithm(
            classifier=self.model.predict, 
            input_vector=input_vector_rev,
            population_size=self.population_size,
            generations=self.generations,
            fitness_function=self._fitness,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            search_space=search_space,
        )
        counterfactual = generator.evolve()
        counterfactual_predictions = self.model.predict(counterfactual)
        counterfactual_probabilities = np.max(self.model.predict_proba([counterfactual]), axis=1)
        return CounterfactualExplanation(
            input_vector=input_vector,
            counterfactuals=counterfactual,
            feature_names=self.feature_names,
            actual_class=instance_class,
            counterfactual_target_class=counterfactual_target_class,
            counterfactual_predictions=counterfactual_predictions,
            counterfactual_probabilities=counterfactual_probabilities,
            graph=None,
        )