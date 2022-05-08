from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: list[str]
    numerical_features: list[str]
    features_to_drop: list[str]
    target_col: Optional[str]