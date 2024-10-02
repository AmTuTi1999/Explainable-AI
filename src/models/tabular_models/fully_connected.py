import torch.nn as nn
from src.models.dataclass import ModelParam

class FullyConnected(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """    
    def __init__(self, model_param: ModelParam, columns: list[str]):
        """_summary_

        Args:
            model_param (_type_): _description_
            columns (_type_): _description_
        """        
        super(FullyConnected, self).__init__()
        self.input_size = len(columns)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, model_param.hidden_layers[0]))
        
        for i in range(1, len(model_param.hidden_layers)):
            self.layers.append(nn.Linear(model_param.hidden_layers[i-1], model_param.hidden_layers[i]))

        self.output_layer = nn.Linear(model_param.hidden_layers[-1], model_param.output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x
