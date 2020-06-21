FROM python:3.7

WORKDIR /app
COPY requirements.txt .
COPY models /app/models

RUN pip install -r requirements.txt

RUN pip install /app/models/pl_political_advertising_model-1.0.0.tar.gz

EXPOSE 8501