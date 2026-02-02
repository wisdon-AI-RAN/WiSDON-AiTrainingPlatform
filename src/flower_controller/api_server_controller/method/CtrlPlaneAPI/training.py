#==========================================================
# Description: Define Ctrl plane API for training AI model. 
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/01/20
# Version: 0.1.0
# License: None
#==========================================================

from typing import List, Optional, Dict
from fastapi import FastAPI, APIRouter, Response, status, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

class ModelInfo(BaseModel):
    project_id: str
    app_name: str
    model_version: str
    timestamp: Optional[str] = None

class TrainingConfig(BaseModel):
    project_id: str
    app_name: str
    model_version: str
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
        Trigger task retrain/delete/complete event 
        """
        # Called by AppPlat to push task to server waiting queue
        @self.router.post('/fl_training_task/push', tags=['Control FL Training Event'])
        async def fl_training_task_push(request: ModelInfo, config: TrainingConfig):
            task_name = request.project_id + "_" + request.app_name + "_" + request.model_version
            try:
                # Connect to MongoDB and save training parameters
                client = MongoClient("mongodb://mongodb:27017")
                database = client["Training_Configuration"]
                collection = database["training_conf"]

                # Delete existing data in MongoDB
                collection.delete_one({
                    "project_id": request.project_id,
                    "app_name": request.app_name,
                    "model_version": request.model_version
                })

                # Insert data into MongoDB
                collection.insert_one(dict(config))

                # Trigger the controller 
                self.controller.task_queue_controller.AiTrPlat.raise_push_in_task_queue()
                self.controller.task_queue_controller.run_cycle()
                self.logger.info(f"Raised retrain event for task: {task_name}")
                return {'task_name': task_name, "action": "retrain task push"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to push training task: {e}")
            
        # Called by AppPlat to delete task while popping out the task from queue
        @self.router.post('/fl_retrain_queue/delete', tags=['Control FL Training Event'])
        async def fl_retrain_queue_delete(request: ModelInfo):
            task_name = request.project_id + "_" + request.app_name + "_" + request.model_version
            if task_name not in self.controller.fl_retrain_delete_list:
                self.controller.fl_retrain_delete_list.append(task_name)
            return {'task_name': task_name, "action": "retrain task delete"}
        
        # Called by AiTrPlat(server) after reinforcement/federated learning retrain finished to clear 'fl retrain task'
        @self.router.post('/fl_retrain_queue/inform_finished', tags=['Control FL Training Event'])
        async def fl_retrain_queue_inform_finished():
            # Remove current retrain symptom since finished
            self.controller.fl_symptom_retrain_task = None
            return {'symptom': self.controller.fl_symptom_retrain_task}