#==========================================================
# Description: Example client for AI Model Repository Server
# Demonstrates how to interact with the model repository API.
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/02/06
# Version: 1.0.0
# License: None
#==========================================================

import requests
import os
from pathlib import Path


class ModelRepositoryClient:
    """Client for interacting with the AI Model Repository Server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check if the server is healthy"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def upload_model(self, file_path: str, project_id: str, app_name: str, model_name: str,
                    version: str, component_name: str, description: str = None, framework: str = None):
        """
        Upload a model to the repository
        
        Args:
            file_path: Path to the model file
            project_id: Project identifier (required)
            app_name: Application name (required)
            model_name: Model name (required)
            version: Model version (required)
            component_name: Component name (required)
            description: Model description (optional)
            framework: AI framework name (optional)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            params = {
                'project_id': project_id,
                'app_name': app_name,
                'model_name': model_name,
                'version': version,
                'component_name': component_name
            }
            
            if description:
                params['description'] = description
            if framework:
                params['framework'] = framework
            
            response = requests.post(
                f"{self.base_url}/models/upload",
                files=files,
                params=params
            )
            response.raise_for_status()
            return response.json()
    
    # def list_models(self, project_id: str = None, app_name: str = None, version: str = None):
    #     """
    #     List models in the repository
        
    #     Args:
    #         project_id: Optional project ID to filter by
    #         app_name: Optional app name to filter by (requires project_id)
    #         version: Optional version to filter by (requires project_id and app_name)
    #     """
    #     if project_id and app_name and version:
    #         url = f"{self.base_url}/models/{project_id}/{app_name}/{version}"
    #     elif project_id and app_name:
    #         url = f"{self.base_url}/models/{project_id}/{app_name}"
    #     elif project_id:
    #         url = f"{self.base_url}/models/{project_id}"
    #     else:
    #         url = f"{self.base_url}/models/list"
        
    #     response = requests.get(url)
    #     response.raise_for_status()
    #     return response.json()
    
    # def get_model_info(self, project_id: str, app_name: str, version: str, model_name: str):
    #     """Get information about a specific model"""
    #     response = requests.get(
    #         f"{self.base_url}/models/{project_id}/{app_name}/{version}/{model_name}/info"
    #     )
    #     response.raise_for_status()
    #     return response.json()
    
    # def download_model(self, project_id: str, app_name: str, version: str, model_name: str,
    #                   save_path: str = None):
    #     """
    #     Download a model from the repository
        
    #     Args:
    #         project_id: Project identifier
    #         app_name: Application name
    #         version: Model version
    #         model_name: Model name
    #         save_path: Path to save the model (default: current directory)
    #     """
    #     response = requests.get(
    #         f"{self.base_url}/models/{project_id}/{app_name}/{version}/{model_name}"
    #     )
    #     response.raise_for_status()
        
    #     if save_path is None:
    #         save_path = f"{project_id}_{app_name}_{version}_{model_name}"
        
    #     with open(save_path, 'wb') as f:
    #         f.write(response.content)
        
    #     return save_path
    
    # def delete_model(self, project_id: str, app_name: str, version: str, model_name: str):
    #     """Delete a model from the repository"""
    #     response = requests.delete(
    #         f"{self.base_url}/models/{project_id}/{app_name}/{version}/{model_name}"
    #     )
    #     response.raise_for_status()
    #     return response.json()


# def main():
#     """Example usage of the ModelRepositoryClient"""
    
#     # Initialize client
#     client = ModelRepositoryClient("http://localhost:8000")
    
#     # Health check
#     print("1. Checking server health...")
#     health = client.health_check()
#     print(f"   Status: {health['status']}")
#     print()
    
#     # Example: Upload a model (you need to have a model file)
#     # Uncomment and modify the path to test
#     """
#     print("2. Uploading a model...")
#     result = client.upload_model(
#         file_path="path/to/your/model.pt",
#         project_id="my_project",
#         app_name="my_app",
#         version="1.0.0",
#         model_name="baseline_model",
#         description="Example PyTorch model",
#         framework="pytorch"
#     )
#     print(f"   Upload result: {result}")
#     print()
#     """
    
#     # List all models
#     print("2. Listing all models...")
#     models = client.list_models()
#     print(f"   Total models: {models['total_models']}")
#     for model in models['models']:
#         print(f"   - {model['project_id']}/{model['app_name']}/{model['version']}/{model['model_name']} ({model['size']} bytes)")
#     print()
    
#     # Get info about a specific model (if any exist)
#     if models['total_models'] > 0:
#         model = models['models'][0]
#         project_id = model['project_id']
#         app_name = model['app_name']
#         version = model['version']
#         model_name = model['model_name']
        
#         print(f"3. Getting info for '{project_id}/{app_name}/{version}/{model_name}'...")
#         info = client.get_model_info(project_id, app_name, version, model_name)
#         print(f"   Info: {info}")
#         print()
        
#         # List models for a specific project
#         print(f"4. Listing models for project '{project_id}'...")
#         project_models = client.list_models(project_id=project_id)
#         print(f"   Total models in project: {project_models['total_models']}")
#         print()
        
#         # Example: Download a model
#         """
#         print(f"5. Downloading '{project_id}/{app_name}/{version}/{model_name}'...")
#         saved_path = client.download_model(project_id, app_name, version, model_name)
#         print(f"   Model saved to: {saved_path}")
#         print()
#         """
        
#         # Example: Delete a model
#         """
#         print(f"6. Deleting '{project_id}/{app_name}/{version}/{model_name}'...")
#         result = client.delete_model(project_id, app_name, version, model_name)
#         print(f"   Delete result: {result}")
#         """


# if __name__ == "__main__":
#     main()
