#==========================================================
# Description: Run AI Model Repository Server Container
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2025/12/24
# Version: 1.0.0
# License: None
#==========================================================

# Configuration
CONTAINER_NAME="ai-model-repository"
IMAGE_NAME="ai-model-repository"
PORT=27016
MODELS_DIR="./models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "AI Model Repository Server"
echo "======================================"

# Create models directory if it doesn't exist
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${YELLOW}Creating models directory...${NC}"
    mkdir -p "$MODELS_DIR"
fi

# Check if container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo -e "${YELLOW}Container is already running.${NC}"
    echo "Container ID: $(docker ps -q -f name=$CONTAINER_NAME)"
    echo -e "\nTo stop: docker stop $CONTAINER_NAME"
    echo -e "To restart: docker restart $CONTAINER_NAME"
    echo -e "To view logs: docker logs -f $CONTAINER_NAME"
    exit 0
fi

# Check if container exists but is stopped
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo -e "${YELLOW}Starting existing container...${NC}"
    docker start $CONTAINER_NAME
    echo -e "${GREEN}Container started!${NC}"
    echo "Access the API at: http://localhost:$PORT"
    echo "API Documentation: http://localhost:$PORT/docs"
    exit 0
fi

# Check if image exists
if [ -z "$(docker images -q $IMAGE_NAME)" ]; then
    echo -e "${YELLOW}Image not found. Building Docker image...${NC}"
    docker build -t $IMAGE_NAME .
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build Docker image${NC}"
        exit 1
    fi
    echo -e "${GREEN}Image built successfully!${NC}"
fi

# Run the container
echo -e "${YELLOW}Starting new container...${NC}"
docker run -d \
    --name $CONTAINER_NAME \
    --network aitrplat \
    -p $PORT:8000 \
    -v "$(pwd)/models:/app/models" \
    -e MODEL_STORAGE_PATH=/app/models \
    -e SERVER_HOST=0.0.0.0 \
    -e SERVER_PORT=8000 \
    --restart unless-stopped \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Container started successfully!${NC}"
    echo ""
    echo "======================================"
    echo "Server Information"
    echo "======================================"
    echo "Container Name: $CONTAINER_NAME"
    echo "API URL: http://localhost:$PORT"
    echo "API Docs: http://localhost:$PORT/docs"
    echo "Health Check: http://localhost:$PORT/health"
    echo "Models Directory: $(pwd)/models"
    echo ""
    echo "======================================"
    echo "Useful Commands"
    echo "======================================"
    echo "View logs: docker logs -f $CONTAINER_NAME"
    echo "Stop server: docker stop $CONTAINER_NAME"
    echo "Restart server: docker restart $CONTAINER_NAME"
    echo "Remove container: docker rm -f $CONTAINER_NAME"
    echo ""
    
    # Wait a moment and check health
    sleep 2
    echo -e "${YELLOW}Checking server health...${NC}"
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is healthy and ready!${NC}"
    else
        echo -e "${YELLOW}⚠ Server is starting... (may take a few seconds)${NC}"
    fi
else
    echo -e "${RED}Failed to start container${NC}"
    exit 1
fi
