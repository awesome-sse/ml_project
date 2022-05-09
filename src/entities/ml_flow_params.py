from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class MlFlowParams:
    use_mlflow: bool = True
    mlflow_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "homework1"