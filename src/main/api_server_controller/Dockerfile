FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=900 -r requirements.txt

# Install curl for healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# # Copy the application code
# COPY . .

ENV API_SERVER_URL="140.113.144.121:9005"

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9005/health || exit 1

# Run the controller
CMD ["python3", "./api_server_controller/runAPI.py"]