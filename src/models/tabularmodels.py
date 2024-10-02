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
        model_param,
    ):
        self.estimator = estimator
        self.model_param = model_param

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
        x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.model_param.batch_size, shuffle=True
        )

        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.model_param.batch_size, shuffle=True
        )
        
        self.estimator(
            train_loader
        )
        self.estimator.compute_loss(
            val_loader
        )

    def evaluate(
        self, 
        test_data: tuple[torch.Tensor, torch.Tensor]
    ):
        """_summary_

        Args:
            test_data (tuple[torch.Tensor, torch.Tensor]): _description_
        """        
        x_test, y_test = test_data
        x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.model_param.batch_size, shuffle=False
        )       
        self.estimator.compute_loss(
            test_loader
        )



class HeartDiseaseNeuralNetwork(TabularNeuralNetworks):
    """_summary_

    Args:
        TabularNeuralNetworks (_type_): _description_
    """    
    def __init__(self, model_param, columns):
        estimator = FullyConnected
        super().__init__(estimator(model_param, columns), model_param)