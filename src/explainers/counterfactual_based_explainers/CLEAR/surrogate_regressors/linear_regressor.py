import pandas as pd
import numpy as np
import logging
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression 
from sklearn.metrics import accuracy_score, recall_score

class LinearRegressor():
    def __init__(
            self, 
            linear_model: LogisticRegression | Lasso | LinearRegression, 
            n_runs: int = 10
        ):
        self.linear_model = linear_model
        self.n_runs = n_runs

    def weights(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y)
        n = 0
        weights = []
        recalls = []
        for _ in range(self.n_runs):
            n += 1
            self.linear_model.fit(x_train, y_train)
            y_pred = self.linear_model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            logging.info(f"Accuracy: {acc}, Recall: {recall}")
            if acc >= 0.8 and recall >= 0.8:
                w = self.linear_model.coef_
                break
            else:
                weight = self.linear_model.coef_
                weights.append(weight)
                recalls.append(recall)
                w = weights[np.argmax(np.array(recalls))] 
        return w[0]