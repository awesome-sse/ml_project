ml in prodaction homework2
==============================

Start server:
------------
    python -m server
------------

Request on server:
------------
    python -m request --PATH_TO_DATA "DATA_PATH" -n 50 -i "localhost" -p 8000
    # -n: Number of rows in file, -i: IP, -p: port
------------

Start server from Docker locally:
------------
    docker build -t aweesomesse/ml_prod_homework2 .
    docker run -p 8000:8000 -it --rm aweesomesse/ml_prod_homework2
------------

Start server from DockerHub:
------------
    docker pull aweesomesse/ml_prod_homework2
    docker run -p 8000:8000 -it --rm aweesomesse/ml_prod_homework2
------------

Start server tests:
------------
    python -m pytest tests\test_server.py
------------

Project structure:
------------
    ├── README.md             <- The top-level README for developers using this project.
    │
    ├── models                <- Trained and serialized models
    │
    ├── requirements.txt      <- The requirements file for reproducing the analysis environment
    │
    │
    ├── entities          <- BaseModels                   
    │   │
    │   └── server_params.py
    │   
    │    
    │
    ├── tests          <- test
    │   │
    │   └── test_server.py  <- Test for /predict
    │  
    ├── server.py       <- Online inference for model
    │
    ├── Dockerfile
    │
    └── request.py <- A script that makes requests to the server
------------