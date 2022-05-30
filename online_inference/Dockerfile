FROM python:3.9

COPY requirements.txt app/requirements.txt
RUN pip3 install -r app/requirements.txt

COPY models app/models
ADD entities app/entities
COPY server.py app/server.py

EXPOSE 8000

WORKDIR /app

ENV PATH_TO_MODEL="models/model.pkl"

CMD python3 -m server