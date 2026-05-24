FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# This ensures that "import app.api" works correctly
ENV PYTHONPATH="/app"

WORKDIR /app

# Install system dependencies for audio
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project folders
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY config/ ./config/
COPY dashboard/ ./dashboard/
COPY training/ ./training/
COPY config/config.yaml ./config.yaml

# Create temp folder for API uploads
RUN mkdir -p /app/temp

# Ports for API and Streamlit
EXPOSE 8000 8501