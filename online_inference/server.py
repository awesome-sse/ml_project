import os
import uvicorn
from fastapi import FastAPI
import logging

from entities import (
    HeartDiseaseModel,
    HeartDiseaseResponse,
    make_predict,
    load_model,
)

app = FastAPI()

logger = logging.getLogger(__name__)

@app.get("/")
def read_root():
    return "homework2"


@app.on_event("startup")
def loading_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_model(model_path)

@app.get("/health")
def read_health():
    return not (model is None)

@app.get("/predict/", response_model=list[HeartDiseaseResponse])
def predict(request: HeartDiseaseModel):
    return make_predict(request.data, request.features, model)

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=os.getenv("PORT", 8000))
