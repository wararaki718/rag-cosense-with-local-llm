FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables will be populated by docker-compose or .env
ENTRYPOINT ["python", "index_data.py"]
