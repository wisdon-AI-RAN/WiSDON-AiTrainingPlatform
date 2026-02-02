FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model storage directory
RUN mkdir -p /app/models

# Copy server code
COPY model_server.py .

# Expose HTTP port
EXPOSE 8000

# Set environment variables
ENV MODEL_STORAGE_PATH=/app/models
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000

# Run the server
CMD ["python", "model_server.py"]
