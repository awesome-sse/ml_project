import sys
import logging
import pickle
import pandas as pd
import click
from src.data import read_data

from src.entities.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_pipeline_params,
)

from src.features import make_features
from src.features.build_features import extract_target, build_transformer
from src.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
    load_model,
)

from src.models.predict_model import create_inference_pipeline

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def predict_pipeline(config_path: str):

    predict_pipeline_params = read_predict_pipeline_params(config_path)
    return run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_pipeline_params):
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    model = load_model(predict_pipeline_params.model_path)

    predicts = predict_model(
        model,
        data,
    )

    pd.DataFrame(predicts).to_csv(predict_pipeline_params.output_predict_path, index=False)

    logger.info(f"predictions are saved in {predict_pipeline_params.output_predict_path}")

    return predict_pipeline_params.output_predict_path


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()