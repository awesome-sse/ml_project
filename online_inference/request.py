import sys
import logging
import requests
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--PATH_TO_DATA", dest="Data_path", type=str,
                    default="data/raw/predict_heart_cleveland_upload.csv")
parser.add_argument("-i", dest="IP", type=str, default="127.0.0.1")
parser.add_argument("-p", dest="Port", type=str, default="8000")
parser.add_argument("-n", dest="Number of request", type=int)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stream_handler)

if __name__ == "__main__":
    data_path = vars(parser.parse_args())["Data_path"]
    ip = vars(parser.parse_args())["IP"]
    port = vars(parser.parse_args())["Port"]
    len_requests = vars(parser.parse_args())["Number of request"]

    logger.info("Read file")

    data = pd.read_csv(data_path)
    data["id"] = data.index
    request_features = list(data.columns)

    len_requests = data.shape[0] if not len_requests else len_requests
    url = "http://" + ip + ":" + port + "/predict/"

    logger.info(f"Number of requests: {len_requests}")
    logger.info(f"Url: {url}")

    for i in range(len_requests):
        request_data = data.iloc[i].tolist()

        logger.info(f"Request data: {request_data}")

        response = requests.get(
            url,
            json={"data": [request_data], "features": request_features}
        )

        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response data: {response.json()}")
