# AI Model Repository Server

A containerized HTTP server for storing and managing AI models using a hierarchical structure based on project, application, version, and model name.

## Features

- **Upload Models**: Save AI models organized by project_id/app_name/version/model_name
- **Download Models**: Retrieve models via HTTP GET
- **List Models**: View all models, or filter by project/application/version
- **Model Metadata**: Store and retrieve model information (framework, description, etc.)
- **Delete Models**: Remove models from the repository
- **Hierarchical Storage**: Models organized as `{project_id}/{app_name}/{version}/{model_name}/`

## Quick Start

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Using Docker

```bash
# Build the image
docker build -t ai-model-repository .

# Run the container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name ai-model-repository \
  ai-model-repository
```

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python model_server.py
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload a Model
```bash
curl -X POST "http://localhost:8000/models/upload?project_id=my_project&app_name=my_app&version=1.0.0&model_name=baseline" \
  -F "file=@my_model.pt" \
  -F "description=My trained model" \
  -F "framework=pytorch"
```

### List All Models
```bash
curl http://localhost:8000/models/list
```

### List Models by Project
```bash
curl http://localhost:8000/models/{project_id}
```

### List Models by Project and App
```bash
curl http://localhost:8000/models/{project_id}/{app_name}
```

### List Models by Project, App and Version
```bash
curl http://localhost:8000/models/{project_id}/{app_name}/{version}
```

### Get Model Info
```bash
curl http://localhost:8000/models/{project_id}/{app_name}/{version}/{model_name}/info
```

### Download a Model
```bash
curl -O http://localhost:8000/models/{project_id}/{app_name}/{version}/{model_name}
# Or with wget
wget http://localhost:8000/models/{project_id}/{app_name}/{version}/{model_name}
```

### Delete a Model
```bash
curl -X DELETE http://localhost:8000/models/{project_id}/{app_name}/{version}/{model_name}
```

## Python Client Example

```python
import requests

# Upload a model
with open('model.pt', 'rb') as f:
    files = {'file': f}
    params = {
        'project_id': 'my_project',
        'app_name': 'my_app',
        'version': '1.0.0',
        'model_name': 'baseline_model',
        'description': 'Trained model',
        'framework': 'pytorch'
    }
    response = requests.post('http://localhost:8000/models/upload', 
                           files=files, params=params)
    print(response.json())

# List all models
response = requests.get('http://localhost:8000/models/list')
print(response.json())

# List models for a specific project
response = requests.get('http://localhost:8000/models/my_project')
print(response.json())

# List models for a specific project and app
response = requests.get('http://localhost:8000/models/my_project/my_app')
print(response.json())

# List models for a specific project, app and version
response = requests.get('http://localhost:8000/models/my_project/my_app/1.0.0')
print(response.json())

# Download a model
response = requests.get('http://localhost:8000/models/my_project/my_app/1.0.0/baseline_model')
with open('downloaded_model.pt', 'wb') as f:
    f.write(response.content)

# Get model info
response = requests.get('http://localhost:8000/models/my_project/my_app/1.0.0/baseline_model/info')
print(response.json())

# Delete a model
response = requests.delete('http://localhost:8000/models/my_project/my_app/1.0.0/baseline_model')
print(response.json())
```

## Configuration

Environment variables:

- `MODEL_STORAGE_PATH`: Directory for storing models (default: `/app/models`)
- `SERVER_HOST`: Server host (default: `0.0.0.0`)
- `SERVER_PORT`: Server port (default: `8000`)
## Storage

Models are stored in a hierarchical structure:
```
./models/
  ├── {project_id}/
  │   ├── {app_name}/
  │   │   ├── {version}/
  │   │   │   ├── {model_name}/
  │   │   │   │   ├── model           # The actual model file
  │   │   │   │   └── metadata.json   # Model metadata
```

This structure is mounted as a volume, ensuring models persist even if the container is removed.

## Security Notes

For production use, consider adding:
- Authentication/Authorization
- HTTPS/TLS encryption
- Rate limiting
- File size limits
- File type validation
