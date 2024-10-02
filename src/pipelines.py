#import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.data.heartdisease.heart_disease import load_data
from src.models.tabularmodels import TabularNeuralNetworks

def init(cfg: DictConfig):
    """_summary_

    Args:
        cfg (DictConfig): _description_

    Returns:
        _type_: _description_
    """    
    model: TabularNeuralNetworks = hydra.utils.instantiate(
        cfg.model, columns= cfg.columns
    )
    (data, target) = load_data(cfg.data)

    return model, (data, target)

def fit_and_evaluate(cfg: DictConfig):
    """_summary_

    Args:
        cfg (DictConfig): _description_
    """    
    model, dataset = init(cfg)
    data, target = dataset

    x_train, val_x, y_train, val_y = train_test_split(data, target, test_size=cfg.val_size)
    x_val, x_test, y_val, y_test = train_test_split(val_x, val_y, test_size=cfg.test_size)

    train_set = (x_train, y_train)
    val_set = (x_val, y_val)
    test_set = (x_test, y_test)

    model.fit(train_set, val_set)
    model.evaluate(test_set)

def cross_validation(cfg: DictConfig):
    """Fit and evaluate the model using K-Fold cross-validation

    Args:
        cfg (DictConfig): Configuration for the experiment including model, data, and cross-validation parameters.
    """
    model, dataset = init(cfg)
    data, target = dataset

    kfold = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    val_scores = []
    test_scores = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(data, target)):
        print(f"Training fold {fold + 1}/{cfg.cv_folds}...")
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]

        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=cfg.test_size)

        train_set = (x_train, y_train)
        val_set = (x_val, y_val)

        model.fit(train_set, val_set)

        val_score = model.evaluate(val_set)
        test_score = model.evaluate((x_test, y_test))

        val_scores.append(val_score)
        test_scores.append(test_score)

        print(f"Fold {fold + 1} - Validation Score: {val_score}, Test Score: {test_score}")

    avg_val_score = np.mean(val_scores)
    avg_test_score = np.mean(test_scores)

    print(f"Average Validation Score: {avg_val_score}")
    print(f"Average Test Score: {avg_test_score}")

    return avg_val_score, avg_test_score




def run_heartdisease_training(cfg: DictConfig):
    """_summary_

    Args:
        cfg (_type_): _description_
    """    
    fit_and_evaluate(cfg)

