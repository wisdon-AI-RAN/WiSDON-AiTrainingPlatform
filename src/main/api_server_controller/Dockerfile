FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=900 -r requirements.txt

# # Copy the application code
# COPY . .

ENV API_SERVER_URL="140.113.144.121:9005"

# Run the controller
CMD ["python3", "./api_server_controller/runAPI.py"]