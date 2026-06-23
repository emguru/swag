FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    INDEXES_DIR=/indexes

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY swag ./swag
COPY data ./data
COPY examples ./examples

EXPOSE 8080

CMD ["uvicorn", "swag.app:app", "--host", "0.0.0.0", "--port", "8080"]
