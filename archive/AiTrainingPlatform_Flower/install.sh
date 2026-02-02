#!/bin/bash

# Function to install and start all services
install() {
    echo "=========================================="
    echo "Installing AI Training Platform Services"
    echo "=========================================="
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if docker compose is available
    if ! docker compose version &> /dev/null; then
        echo "Error: docker compose (v2) is not installed or not available as a plugin. Please install docker compose v2."
        exit 1
    fi
    
    # Create aitrplat network if it doesn't exist
    if ! docker network inspect aitrplat &> /dev/null; then
        echo "Creating aitrplat network..."
        docker network create aitrplat
    else
        echo "Network aitrplat already exists."
    fi
    
    # Build and start all services
    echo "Building and starting services..."
    docker compose up -d --build
    
    echo ""
    echo "=========================================="
    echo "Installation completed!"
    echo "=========================================="
    echo "Services running:"
    docker compose ps
    echo ""
    echo "To view logs: docker compose logs -f [service-name]"
    echo "To stop all: ./install.sh uninstall"
}

# Function to uninstall and stop all services
uninstall() {
    echo "=========================================="
    echo "Uninstalling AI Training Platform Services"
    echo "=========================================="
    
    # Stop and remove all containers
    echo "Stopping and removing containers..."
    docker compose down
    
    echo ""
    echo "=========================================="
    echo "Uninstallation completed!"
    echo "=========================================="
    echo "Note: Volumes are preserved. To remove volumes, run:"
    echo "  docker compose down -v"
}

# Main script execution
case "$1" in
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Usage: $0 {install|uninstall}"
        echo ""
        echo "Commands:"
        echo "  install     - Build and start all services"
        echo "  uninstall   - Stop and remove all services"
        exit 1
        ;;
esac
