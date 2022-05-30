FROM python:3.9

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD models models
ADD entities entities
COPY server.py server.py

WORKDIR .

ENV PATH_TO_MODEL="../models/model.pkl"

CMD python ./server.py