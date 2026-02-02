#==========================================================
# Description: Define Ctrl plane API for training AI model. 
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/01/20
# Version: 0.1.0
# License: None
#==========================================================

import sys, os
from typing import List, Optional, Dict
from fastapi import FastAPI, APIRouter, Response, status, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

class ModelInfo(BaseModel):
    project_id: str
    app_name: str
    model_version: str
    mode: str
    timestamp: str

class TrainingConfig(BaseModel):
    project_id: str
    app_name: str
    model_version: str
    mode: str
    epochs: int
    learning_rate: float
    timestamp: str

class TrainingAPI:
    """
    API for training AI model.
    """
    def __init__(self, controller, task_queue, logger):
        
        self.router = APIRouter()
        self.controller = controller
        self.task_queue = task_queue
        self.logger = logger
        """
        Trigger task training/delete/complete event 
        """
        # Called by AppPlat to push task to server waiting queue
        @self.router.post('/fl_training_task/push', tags=['Control FL Training Event'])
        async def fl_training_task_push(request: ModelInfo, config: TrainingConfig):
            try:
                # Connect to MongoDB and save training parameters
                mongodb_url = os.environ.get("MONGODB_URL", "mongodb://mongodb:27017")
                client = MongoClient(mongodb_url)
                database = client["Training_Configuration"]
                collection = database["training_conf"]

                # Insert data into MongoDB
                insert_data = {
                    "project_id": request.project_id,
                    "app_name": request.app_name,
                    "model_version": request.model_version,
                    "mode": request.mode,
                    "status": "waiting",
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "timestamp": request.timestamp
                }
                collection.insert_one(insert_data)

                # Trigger the controller 
                self.controller.task_name = {
                    "project_id": request.project_id,
                    "app_name": request.app_name,
                    "model_version": request.model_version,
                    "mode": request.mode,
                }
                self.controller.task_queue_controller.ai_tr_plat.raise_push_in_task_queue()
                self.controller.task_queue_controller.run_cycle()
                self.logger.info(f"Raise training event:\n{request.project_id}-{request.app_name}-{request.model_version} in {request.mode} mode")
                
                return {
                    'project_id': request.project_id,
                    'app_name': request.app_name,
                    'model_version': request.model_version,
                    "mode": request.mode,
                    "action": "Start training task"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to push training task: {e}")
            
        # Called by AppPlat to delete task while popping out the task from queue
        @self.router.post('/fl_training_task/delete', tags=['Control FL Training Event'])
        async def fl_training_task_delete(request: ModelInfo):
            try:
                # Connect to MongoDB and save training parameters
                mongodb_url = os.environ.get("MONGODB_URL", "mongodb://mongodb:27017")
                client = MongoClient(mongodb_url)
                database = client["Training_Configuration"]
                collection = database["training_conf"]

                # Delete existing data in MongoDB
                collection.delete_one({
                    "project_id": request.project_id,
                    "app_name": request.app_name,
                    "model_version": request.model_version,
                    "mode": request.mode
                })

                # Trigger the controller 
                self.controller.delete_name = {
                    "project_id": request.project_id,
                    "app_name": request.app_name,
                    "model_version": request.model_version,
                    "mode": request.mode,
                }
                self.controller.task_queue_controller.ai_tr_plat.raise_delete_task_in_task_queue()
                self.controller.task_queue_controller.run_cycle()
                self.logger.info(f"Raise deleting event:\n{request.project_id}-{request.app_name}-{request.model_version} in {request.mode} mode")

                return {
                    'project_id': request.project_id,
                    'app_name': request.app_name,
                    'model_version': request.model_version,
                    "mode": request.mode,
                    "action": "Delete training task"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to delete training task: {e}")
                
        # Show task status
        @self.router.get('/fl_training_task/show_task_status/{project_id}/{app_name}/{model_version}/{mode}', tags=['Control FL Training Event'])
        async def fl_training_task_show_status(project_id: str, app_name: str, model_version: str, mode: str):
            try:
                task_name = {
                    "project_id": project_id,
                    "app_name": app_name,
                    "model_version": model_version,
                    "mode": mode,
                }
                report = self.task_queue.is_duplicate_report(task_name)
                if report:
                    if self.task_queue.fl_current_training_task == task_name:
                        status = "running"
                    else:
                        status = "waiting"
                else:
                    status = "not in queue"
                return {'project_id': project_id, 'app_name': app_name, 'model_version': model_version, "mode": mode, 'status': status}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to show training task status: {e}")