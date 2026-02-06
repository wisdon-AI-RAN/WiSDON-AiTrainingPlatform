#==========================================================
# Description: Define Ctrl plane API for training AI model. 
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/02/06
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
    model_name: str
    model_version: str
    mode: str
    dataset_name: str
    timestamp: str

class TrainingConfig(BaseModel):
    project_id: str
    app_name: str
    model_name: str
    model_version: str
    mode: str
    dataset_name: str
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
                mongodb_url = os.environ.get("AITRCOMMONDB_URI")
                if not mongodb_url:
                    raise HTTPException(status_code=500, detail="[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to connect to MongoDB: AITRCOMMONDB_URI not found in environment variables")
                client = MongoClient(mongodb_url)
                database = client["TrainingConfig"]
                collection_name = f"{config.project_id}_{config.app_name}_{config.model_name}_{config.model_version}_{config.mode}_{config.dataset_name}"
                collection = database[collection_name]

                filter_query = {
                    "project_id": config.project_id,
                    "app_name": config.app_name,
                    "model_name": config.model_name,
                    "model_version": config.model_version,
                    "mode": config.mode,
                    "dataset_name": config.dataset_name
                }

                # if existing data, delete it first
                if collection.count_documents(filter_query) > 0:
                    collection.delete_many(filter_query)

                # Insert data into MongoDB
                insert_data = {
                    "project_id": config.project_id,
                    "app_name": config.app_name,
                    "model_name": config.model_name,
                    "model_version": config.model_version,
                    "mode": config.mode,
                    "dataset_name": config.dataset_name,
                    "status": "waiting",
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "timestamp": config.timestamp
                }
                collection.insert_one(insert_data)

                # Trigger the controller 
                self.controller.task_name = {
                    "project_id": config.project_id,
                    "app_name": config.app_name,
                    "model_name": config.model_name,
                    "model_version": config.model_version,
                    "mode": config.mode,
                    "dataset_name": config.dataset_name
                }
                self.controller.task_queue_controller.ai_tr_plat.raise_push_in_task_queue()
                # self.controller.task_queue_controller.run_cycle()
                self.logger.info(f"Raise training event:\n{config.project_id}-{config.app_name}-{config.model_name}-{config.model_version} in {config.mode} mode")
                self.logger.info(f"Use {config.dataset_name} dataset, epochs: {config.epochs}, learning_rate: {config.learning_rate}")
                return {
                    'project_id': config.project_id,
                    'app_name': config.app_name,
                    "model_name": config.model_name,
                    'model_version': config.model_version,
                    "mode": config.mode,
                    "dataset_name": config.dataset_name,
                    "action": "Start training task"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to push training task: {e}")
            
        # Called by AppPlat to delete task while popping out the task from queue
        @self.router.post('/fl_training_task/delete', tags=['Control FL Training Event'])
        async def fl_training_task_delete(request: ModelInfo):
            try:
                # Connect to MongoDB and save training parameters
                mongodb_url = os.environ.get("AITRCOMMONDB_URI")
                if not mongodb_url:
                    raise HTTPException(status_code=500, detail="[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to connect to MongoDB: AITRCOMMONDB_URI not found in environment variables")
                client = MongoClient(mongodb_url)
                database = client["TrainingConfig"]
                collection_name = f"{request.project_id}_{request.app_name}_{request.model_name}_{request.model_version}_{request.mode}_{request.dataset_name}"
                collection = database[collection_name]

                # Delete existing data in MongoDB
                collection.delete_one({
                    "project_id": request.project_id,
                    "app_name": request.app_name,
                    "model_name": request.model_name,
                    "model_version": request.model_version,
                    "mode": request.mode,
                    "dataset_name": request.dataset_name
                })

                # Trigger the controller 
                self.controller.delete_name = {
                    "project_id": request.project_id,
                    "app_name": request.app_name,
                    "model_name": request.model_name,
                    "model_version": request.model_version,
                    "mode": request.mode,
                    "dataset_name": request.dataset_name
                }
                self.controller.task_queue_controller.ai_tr_plat.raise_delete_task_in_task_queue()
                # self.controller.task_queue_controller.run_cycle()
                self.logger.info(f"Raise deleting event:\n{request.project_id}-{request.app_name}-{request.model_name}-{request.model_version} in {request.mode} mode")

                return {
                    'project_id': request.project_id,
                    'app_name': request.app_name,
                    "model_name": request.model_name,
                    'model_version': request.model_version,
                    "mode": request.mode,
                    "dataset_name": request.dataset_name,
                    "action": "Delete training task"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to delete training task: {e}")
                
        # Show task status
        @self.router.get('/fl_training_task/show_task_status/{project_id}/{app_name}/{model_name}/{model_version}/{mode}/{dataset_name}', tags=['Control FL Training Event'])
        async def fl_training_task_show_status(project_id: str, app_name: str, model_name: str, model_version: str, mode: str, dataset_name: str):
            try:
                task_name = {
                    "project_id": project_id,
                    "app_name": app_name,
                    "model_name": model_name,
                    "model_version": model_version,
                    "mode": mode,
                    "dataset_name": dataset_name
                }

                report = self.task_queue.is_duplicate_report(task_name)
                if report:
                    if self.task_queue.fl_current_training_task == task_name:
                        status = "running"
                    else:
                        status = "waiting"
                else:
                    status = "not in queue"

                return {'project_id': project_id, 'app_name': app_name, 'model_name': model_name, 'model_version': model_version, "mode": mode, "dataset_name": dataset_name, 'status': status}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to show training task status: {e}")