import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

SklearnRegressionModel = Union[RandomForestRegressor, LinearRegression]


def predict_model(
    model: Pipeline, features: pd.DataFrame, use_log_trick: bool = False
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy_score": accuracy_score(target, predicts),
        "precision_score": precision_score(target, predicts),
        "recall_score": recall_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def create_inference_pipeline(
    model: SklearnRegressionModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output

