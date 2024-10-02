from dataclasses import dataclass

@dataclass
class ModelParam:
    outputsize: int
    hidden_layers: list[int]
    output_size: int

