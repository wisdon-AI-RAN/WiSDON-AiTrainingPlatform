#==========================================================
# Description: Define Ctrl plane API for task queue, only called by AiTrPlat(server). 
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
    mode: str
    timestamp: str

class TaskQueueAPI:
    """
    API for task queuing, only called by AiTrPlat(server).
    """
    def __init__(self, controller, task_queue, logger):
        
        self.router = APIRouter()
        self.controller = controller
        self.task_queue = task_queue
        self.logger = logger
        """
        Trigger task retrain/delete/complete event 
        """
        # Called by AiTrPlat to push task to server waiting queue
        @self.router.post('/fl_training_task_queue/push', tags=['Control Task Queue'])
        async def fl_training_task_push():
            task_name = self.controller.task_name["project_id"] + "_" + self.controller.task_name["app_name"] + "_" + self.controller.task_name["model_version"]
            try:
                # Trigger the controller 
                self.task_queue.fl_training_queue_push(task_name, "0")
                self.controller.task_queue_controller.run_cycle()
            except Exception as e:
                self.controller.task_queue_controller.run_cycle()
                raise HTTPException(status_code=500, detail=f"Error Code 100 [CONNECTION_ERROR]: Failed to push task: {e}")
            
        # Called by AiTrPlat to run the task queue cycle
        @self.router.post('/fl_training_task_queue/run_cycle', tags=['Control Task Queue'])
        async def fl_training_task_run_cycle():
            try:
                self.controller.task_queue_controller.run_cycle()
            except Exception as e:
                self.controller.task_queue_controller.run_cycle()
                raise HTTPException(status_code=500, detail=f"Error Code 100 [CONNECTION_ERROR]: Failed to run task queue controller cycle: {e}")