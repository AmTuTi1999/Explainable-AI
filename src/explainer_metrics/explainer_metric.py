from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ExplainerMetric:
    _name: str
    _direction: str
    _calculated_metrics: list[float]
    
    def __init__(self) -> None:
        super().__init__()



    @abstractmethod
    def init_explainer_metric(
        self,
        model,
        data_batch,
        explanations,
    ) -> None:
        pass