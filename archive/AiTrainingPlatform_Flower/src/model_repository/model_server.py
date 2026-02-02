#==========================================================
# Description: AI Model Repository Server
# HTTP server for saving and loading AI models.
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2025/12/24
# Version: 1.0.0
# License: None
#==========================================================

import os
import json
import shutil
import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Configuration
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "/app/models")
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Ensure storage directory exists
Path(MODEL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="AI Model Repository Server",
    description="HTTP API for storing and retrieving AI models",
    version="1.0.0"
)


def get_model_path(project_id: str, app_name: str, version: str, model_name: str) -> tuple:
    """
    Get the file paths for a model
    Returns: (model_directory, model_file_path, metadata_path)
    """
    model_dir = os.path.join(MODEL_STORAGE_PATH, project_id, app_name, version, model_name)
    model_file = os.path.join(model_dir, "model.onnx")
    metadata_file = os.path.join(model_dir, "metadata.json")
    return model_dir, model_file, metadata_file

def get_model_metadata(project_id: str, app_name: str, version: str, model_name: str) -> dict:
    """Get metadata for a model"""
    model_dir, model_path, metadata_path = get_model_path(project_id, app_name, version, model_name)
    
    metadata = {
        "project_id": project_id,
        "app_name": app_name,
        "version": version,
        "model_name": model_name,
        "exists": os.path.exists(model_path),
        "size": 0,
        "created_at": None,
        "modified_at": None
    }
    
    if os.path.exists(model_path):
        stat = os.stat(model_path)
        metadata["size"] = stat.st_size
        metadata["created_at"] = datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
        metadata["modified_at"] = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
    
    # Load additional metadata if exists
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                extra_metadata = json.load(f)
                metadata.update(extra_metadata)
        except Exception:
            pass
    
    return metadata


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AI Model Repository Server",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /models/upload?project_id=<id>&app_name=<name>&version=<ver>&model_name=<name>",
            "download": "GET /models/{project_id}/{app_name}/{version}/{model_name}",
            "list": "GET /models/list (or /models/{project_id} or /models/{project_id}/{app_name} or /models/{project_id}/{app_name}/{version})",
            "delete": "DELETE /models/{project_id}/{app_name}/{version}/{model_name}",
            "info": "GET /models/{project_id}/{app_name}/{version}/{model_name}/info",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "storage_path": MODEL_STORAGE_PATH
    }


@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    project_id: str = Query(..., description="Project ID"),
    app_name: str = Query(..., description="Application name"),
    version: str = Query(..., description="Model version"),
    model_name: str = Query(..., description="Model name"),
    description: Optional[str] = Query(None, description="Model description"),
    framework: Optional[str] = Query(None, description="AI framework (pytorch, tensorflow, etc.)")
):
    """
    Upload an AI model file
    
    - **file**: Model file to upload
    - **project_id**: Project identifier (required)
    - **app_name**: Application name (required)
    - **version**: Model version (required)
    - **model_name**: Model name (required)
    - **description**: Optional description
    - **framework**: Optional framework name
    """
    try:
        # Get model paths
        model_dir, model_path, metadata_path = get_model_path(project_id, app_name, version, model_name)
        
        # Create directory structure
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded file
        content = await file.read()
        with open(model_path, "wb") as f:
            f.write(content)
        
        # Save metadata
        metadata = {
            "project_id": project_id,
            "app_name": app_name,
            "version": version,
            "model_name": model_name,
            "original_filename": file.filename,
            "content_type": file.content_type,
            "upload_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "size": len(content)
        }
        
        if description:
            metadata["description"] = description
        if framework:
            metadata["framework"] = framework
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return JSONResponse(
            status_code=201,
            content={
                "message": "Model uploaded successfully",
                "project_id": project_id,
                "app_name": app_name,
                "version": version,
                "model_name": model_name,
                "size": len(content),
                "path": model_path
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/models/{project_id}/{app_name}/{version}/{model_name}")
async def download_model(project_id: str, app_name: str, version: str, model_name: str):
    """
    Download a model file
    
    - **project_id**: Project identifier
    - **app_name**: Application name
    - **version**: Model version
    - **model_name**: Model name
    """
    model_dir, model_path, metadata_path = get_model_path(project_id, app_name, version, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model not found: project_id='{project_id}', app_name='{app_name}', version='{version}', model_name='{model_name}'"
        )
    
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    # Generate a meaningful filename for download
    download_filename = f"{project_id}_{app_name}_{version}_{model_name}"
    
    return FileResponse(
        path=model_path,
        filename=download_filename,
        media_type="application/octet-stream"
    )


@app.get("/models/list")
async def list_all_models():
    """
    List all available models in the repository
    """
    try:
        models = []
        
        # Walk through the storage directory structure
        if os.path.exists(MODEL_STORAGE_PATH):
            for project_id in os.listdir(MODEL_STORAGE_PATH):
                project_path = os.path.join(MODEL_STORAGE_PATH, project_id)
                if not os.path.isdir(project_path):
                    continue
                
                for app_name in os.listdir(project_path):
                    app_path = os.path.join(project_path, app_name)
                    if not os.path.isdir(app_path):
                        continue
                    
                    for version in os.listdir(app_path):
                        version_path = os.path.join(app_path, version)
                        if not os.path.isdir(version_path):
                            continue
                        
                        for model_name in os.listdir(version_path):
                            model_name_path = os.path.join(version_path, model_name)
                            if not os.path.isdir(model_name_path):
                                continue
                            
                            model_file = os.path.join(model_name_path, "model.onnx")
                            if os.path.exists(model_file):
                                metadata = get_model_metadata(project_id, app_name, version, model_name)
                                models.append(metadata)
        
        return {
            "total_models": len(models),
            "models": models
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{project_id}")
async def list_project_models(project_id: str):
    """
    List all models for a specific project
    
    - **project_id**: Project identifier
    """
    try:
        models = []
        project_path = os.path.join(MODEL_STORAGE_PATH, project_id)
        
        if not os.path.exists(project_path):
            return {
                "project_id": project_id,
                "total_models": 0,
                "models": []
            }
        
        for app_name in os.listdir(project_path):
            app_path = os.path.join(project_path, app_name)
            if not os.path.isdir(app_path):
                continue
            
            for version in os.listdir(app_path):
                version_path = os.path.join(app_path, version)
                if not os.path.isdir(version_path):
                    continue
                
                for model_name in os.listdir(version_path):
                    model_name_path = os.path.join(version_path, model_name)
                    if not os.path.isdir(model_name_path):
                        continue
                    
                    model_file = os.path.join(model_name_path, "model.onnx")
                    if os.path.exists(model_file):
                        metadata = get_model_metadata(project_id, app_name, version, model_name)
                        models.append(metadata)
        
        return {
            "project_id": project_id,
            "total_models": len(models),
            "models": models
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{project_id}/{app_name}")
async def list_app_models(project_id: str, app_name: str):
    """
    List all versions for a specific project and app
    
    - **project_id**: Project identifier
    - **app_name**: Application name
    """
    try:
        models = []
        app_path = os.path.join(MODEL_STORAGE_PATH, project_id, app_name)
        
        if not os.path.exists(app_path):
            return {
                "project_id": project_id,
                "app_name": app_name,
                "total_models": 0,
                "models": []
            }
        
        for version in os.listdir(app_path):
            version_path = os.path.join(app_path, version)
            if not os.path.isdir(version_path):
                continue
            
            for model_name in os.listdir(version_path):
                model_name_path = os.path.join(version_path, model_name)
                if not os.path.isdir(model_name_path):
                    continue
                
                model_file = os.path.join(model_name_path, "model.onnx")
                if os.path.exists(model_file):
                    metadata = get_model_metadata(project_id, app_name, version, model_name)
                    models.append(metadata)
        
        return {
            "project_id": project_id,
            "app_name": app_name,
            "total_models": len(models),
            "models": models
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{project_id}/{app_name}/{version}")
async def list_version_models(project_id: str, app_name: str, version: str):
    """
    List all models for a specific project, app and version
    
    - **project_id**: Project identifier
    - **app_name**: Application name
    - **version**: Model version
    """
    try:
        models = []
        version_path = os.path.join(MODEL_STORAGE_PATH, project_id, app_name, version)
        
        if not os.path.exists(version_path):
            return {
                "project_id": project_id,
                "app_name": app_name,
                "version": version,
                "total_models": 0,
                "models": []
            }
        
        for model_name in os.listdir(version_path):
            model_name_path = os.path.join(version_path, model_name)
            if not os.path.isdir(model_name_path):
                continue
            
            model_file = os.path.join(model_name_path, "model.onnx")
            if os.path.exists(model_file):
                metadata = get_model_metadata(project_id, app_name, version, model_name)
                models.append(metadata)
        
        return {
            "project_id": project_id,
            "app_name": app_name,
            "version": version,
            "total_models": len(models),
            "models": models
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{project_id}/{app_name}/{version}/{model_name}/info")
async def get_model_info(project_id: str, app_name: str, version: str, model_name: str):
    """
    Get detailed information about a specific model
    
    - **project_id**: Project identifier
    - **app_name**: Application name
    - **version**: Model version
    - **model_name**: Model name
    """
    model_dir, model_path, metadata_path = get_model_path(project_id, app_name, version, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: project_id='{project_id}', app_name='{app_name}', version='{version}', model_name='{model_name}'"
        )
    
    metadata = get_model_metadata(project_id, app_name, version, model_name)
    return metadata


@app.delete("/models/{project_id}/{app_name}/{version}/{model_name}")
async def delete_model(project_id: str, app_name: str, version: str, model_name: str):
    """
    Delete a model from the repository
    
    - **project_id**: Project identifier
    - **app_name**: Application name
    - **version**: Model version
    - **model_name**: Model name
    """
    model_dir, model_path, metadata_path = get_model_path(project_id, app_name, version, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: project_id='{project_id}', app_name='{app_name}', version='{version}', model_name='{model_name}'"
        )
    
    try:
        # Delete entire model directory
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        return {
            "message": "Model deleted successfully",
            "project_id": project_id,
            "app_name": app_name,
            "version": version,
            "model_name": model_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


if __name__ == "__main__":
    print(f"Starting AI Model Repository Server")
    print(f"Storage Path: {MODEL_STORAGE_PATH}")
    print(f"Server: http://{SERVER_HOST}:{SERVER_PORT}")
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )
