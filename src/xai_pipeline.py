from omegaconf import DictConfig
import torch
import hydra
from pandas import DataFrame
import logging
from numpy.typing import NDArray
from tqdm import tqdm   

from src.models.tabularmodels import TabularNeuralNetworks
from src.explainers.counterfactual_based_explainers.counterfactual_explainer_base import CounterfactualExplainerBase
from src.explainers.explainer_wrapper import CounterfactualExplainerWrapper 
from src.explainers.counterfactual_explanation import CounterfactualExplanation
from src.pipelines import init
from src.explainers.counterfactual_based_explainers.CLEAR.clear import CLEAR
from src.explainer_metrics.explainer_metric import ExplainerMetric
from src.utils import make_attributions_into_array, make_attrributes_individual, make_x_batch_and_y_batch

def load_model_and_data(cfg: DictConfig, model: torch.nn.Module, data: tuple[DataFrame, DataFrame]) -> tuple[torch.nn.Module, tuple[DataFrame, DataFrame]]:
    """
    Load the model and data based on the configuration.

    Args:
        cfg: Configuration object containing model and data parameters.
        model: The model to be loaded.

    Returns:
        Tuple of loaded model and data.
    """
    if isinstance(model, TabularNeuralNetworks):
        model.load_model_from_dict_state()
    
    return model, data

def initialize_counterfactual_explainers(
        cfg: DictConfig,
        model: torch.nn.Module,
        data: tuple[DataFrame, DataFrame],
) -> dict[str, CounterfactualExplainerWrapper]:
    """
    Generate explanations for the given model and data loader using the specified explainer.

    Args:
        model: The model to explain.
        data_loader: DataLoader providing the input data.
        explainer: The explainer object to generate explanations.
        device: Device to run the model on (CPU or GPU).
        num_samples: Number of samples to explain. If None, all samples will be used.
        batch_size: Batch size for processing.

    Returns:
        List of explanations.
    """

    explainer_wrappers = {}
    explainer_funcs = {}
    x_batch, y_batch = data
    logging.info("Instantiating Explainer Wrappers...")
    for explainer_name in cfg.explainers:
        explainer: CounterfactualExplainerBase = hydra.utils.instantiate(
            cfg.explainers[explainer_name],
            feature_names=cfg.data.data.columns,
            immutable_features=cfg.immutable_features,
            categorical_features=cfg.data.data.categorical_columns,
        )
        logging.info(f"Initializing {explainer.alias()} explainer...")
        explainer.init_explainer(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,    
        )
        explainer_wrapper: CounterfactualExplainerWrapper = CounterfactualExplainerWrapper(
            explainer=explainer,
            num_samples=cfg.num_samples,
            counterfactual_target_class=cfg.counterfactual_target_class,
        )
        explainer_wrappers[explainer.alias()] = explainer_wrapper
        explainer_funcs[explainer.alias()] = explainer_wrapper.explainer_func

    return explainer_wrappers, explainer_funcs
    

def generate_counterfactuals(
        cfg: DictConfig,
        model: torch.nn.Module,
        data: tuple[DataFrame, DataFrame],
        explainer_wrappers: dict,
) -> dict[str, CounterfactualExplanation]:
    """
    Generate counterfactuals for the given model and data loader using the specified explainer.

    Args:
        model: The model to explain.
        data_loader: DataLoader providing the input data.
        explainer: The explainer object to generate explanations.
        device: Device to run the model on (CPU or GPU).
        num_samples: Number of samples to explain. If None, all samples will be used.
        batch_size: Batch size for processing.

    Returns:
        List of explanations.
    """
    counterfactual_explanations_dict = {}
    for explainer_alias, explainer_wrapper in explainer_wrappers.items():
        logging.info(f"Generating counterfactuals using {explainer_alias} for {cfg.num_samples} samples...")
        counterfactual_explanations = explainer_wrapper.explain_local(data[0])

        counterfactual_explanations_dict[explainer_alias] = counterfactual_explanations
    logging.info("Counterfactuals generated successfully.")
    return counterfactual_explanations_dict

def evaluate_explanations(
        cfg: DictConfig,
        model: torch.nn.Module,
        data,
        counterfactual_explanations: dict[str, NDArray],
        explainer_funcs: dict[str, callable],
) -> dict[str, dict[str, float]]:
    """
    Evaluate the generated counterfactuals.ydra.utils.instantiate(
            cfg.explainer_metrics[explainer_metric],
        )
        metric_values = {}
        for explainer_name, explainer_values in counterfactual_explanations.items():
            explainer_metric.ini

    Args:
        cfg: Configuration object containing model and data parameters.
        model: The model to evaluate.
        data: DataLoader providing the input data.
        counterfactual_explanations: Dictionary of counterfactual explanations.

    Returns:
        Dictionary of evaluation metrics.
    """
    logging.info("Instantiating Explainer Metrics...")
    all_metrics = {}
    for explainer_metric in cfg.explainer_metrics:
        explainer_metric: ExplainerMetric = hydra.utils.instantiate(
            cfg.explainer_metrics[explainer_metric],
        )
        metric_values = {}
        for explainer_name, explainer_values in tqdm(counterfactual_explanations.items()):
            explainer_metric.init_explainer_metric(
                model=model,
                data_batch=data,
                explanations=explainer_values,    
                explainer_func=explainer_funcs[explainer_name],
            )

            metric_values[explainer_name] = explainer_metric._calculated_metrics
        all_metrics[explainer_metric] = metric_values
    logging.info("Evaluation metrics calculated successfully.")
    return all_metrics

def run_explain_pipeline(
        cfg: DictConfig,
        model: torch.nn.Module,
        data: tuple[DataFrame, DataFrame],
    ):
    """
    Run the explain pipeline.

    Args:
        cfg: Configuration object containing model and data parameters.
    """
    model, data = load_model_and_data(cfg, model=model, data=data)
    explainer_wrappers, explainer_funcs = initialize_counterfactual_explainers(cfg, model=model, data=data)
    counterfactual_explanations = generate_counterfactuals(cfg, model, data, explainer_wrappers)
    attributions_array = make_attributions_into_array(counterfactual_explanations)
    explanation_metrics = evaluate_explanations(cfg, model.estimator, data, counterfactual_explanations=attributions_array, explainer_funcs=explainer_funcs)
    print(explanation_metrics)
    return counterfactual_explanations

def explain_pipeline(cfg: DictConfig):
    """
    Run the explain pipeline.

    Args:
        cfg: Configuration object containing model and data parameters.
    """
    model, data = init(cfg)
    run_explain_pipeline(cfg, model=model, data=data)