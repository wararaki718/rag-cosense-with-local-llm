FROM python:3.11-slim

WORKDIR /app

# Install dependencies for pandas and other libs
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables with defaults for Docker network
ENV ELASTICSEARCH_URL=http://elasticsearch:9200
ENV APP_API_URL=http://app-api:8000/query
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV LLM_MODEL=gemma2:9b

# Default command (can be overridden to run dataset_generator.py)
CMD ["python", "evaluate.py"]
