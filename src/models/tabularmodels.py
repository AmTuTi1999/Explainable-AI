import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.tabular_models.tabular_model import TabularClassifiers
from src.models.tabular_models.fully_connected import FullyConnected

class TabularNeuralNetworks:
    """_summary_
    """
    def __init__(
        self, 
        estimator: TabularClassifiers,
    ):
        self.estimator = estimator
        self.model_param = estimator.model_param
        self.binary_classifier =estimator.model_param.binary_classifier

    def __call__(self, x):
        """_summary_
        """
        return self.estimator.model(x).detach()
    
    def fit(
        self, 
        train_data: tuple[torch.Tensor, torch.Tensor], 
        val_data: tuple[torch.Tensor, torch.Tensor] 
    ):
        """_summary_

        Args:
            train_data (tuple[torch.Tensor, torch.Tensor]): _description_
        """        
        x_train, y_train = train_data
        x_val, y_val = val_data
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.model_param.batch_size, shuffle=True
        )

        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.model_param.batch_size, shuffle=True
        )
        
        self.estimator.fit(
            train_loader
        )
        self.estimator.compute_loss(
            val_loader, dataset_name="Validation",
        )
        

    def load_model_from_dict_state(self):         
        """_summary_

        Args:
            path (str): _description_
        """        
        self.estimator.load_model_from_dict_state()

    def evaluate(
        self, 
        test_data: tuple[torch.Tensor, torch.Tensor]
    ):
        """_summary_

        Args:
            test_data (tuple[torch.Tensor, torch.Tensor]): _description_
        """        
        x_test, y_test = test_data
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.model_param.batch_size, shuffle=False
        )       
        self.estimator.compute_loss(
            test_loader
        )

    def predict(
        self, 
        test_data: torch.Tensor
    ):
        import pandas as pd
        import numpy as np
        if isinstance(test_data, pd.DataFrame) | isinstance(test_data, pd.Series) | isinstance(test_data, torch.Tensor):
            test_data = test_data.values
        if isinstance(test_data, np.ndarray):
            test_data = torch.tensor(test_data, dtype=torch.float32)
        """_summary_

        Args:
            test_data (torch.Tensor): _description_
        """        
        if test_data.ndim == 1:  # If single vector, add batch dimension
            test_data = test_data.unsqueeze(0)
        x_test_tensor = test_data.clone().detach()
        predictions = self.estimator.model(x_test_tensor).detach().numpy()
        if self.binary_classifier:
            predictions = (predictions > 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(predictions, axis=1)
        return predictions
    
    def predict_proba_single(
        self, 
        test_data: torch.Tensor
    ):
        """_summary_

        Args:
            test_data (torch.Tensor): _description_
        """        
        import numpy as np
        predict_probabilities = [0,0]
        predictions = self.estimator.model(torch.tensor(test_data, dtype=torch.float32)).detach().numpy()[0]
        if self.binary_classifier:
            index = 1 if predictions > self.model_param.threshold else 0
            predict_probabilities[index], predict_probabilities[1 - index] = predictions, 1 - predictions
        else:
            predict_probabilities = predictions
        return np.array(predict_probabilities)


    def predict_proba(self, test_data):
        if self.binary_classifier:
            return [self.predict_proba_single(test_data[i]) for i in range(len(test_data))]
        else:
            return self.estimator.model(test_data).detach().numpy()
    
class HeartDiseaseNeuralNetwork(TabularNeuralNetworks):
    """_summary_

    Args:
        TabularNeuralNetworks (_type_): _description_
    """    
    def __init__(self, model_param, columns):
        estimator = FullyConnected
        super().__init__(TabularClassifiers(model=estimator, model_param=model_param, columns=columns))