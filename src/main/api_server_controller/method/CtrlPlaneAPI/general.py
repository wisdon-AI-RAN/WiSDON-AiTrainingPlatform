#==========================================================
# Description: Define general Ctrl plane API. 
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/01/20
# Version: 0.1.0
# License: None
#==========================================================

from typing import List, Optional, Dict
from fastapi import FastAPI, APIRouter, Response, status, HTTPException
from pydantic import BaseModel
import threading

class ModelInfo(BaseModel):
    project_id: str
    app_name: str
    model_version: str
    mode: str
    timestamp: str

class GeneralCtrlAPI:
    """
    Define general control plane API.
    """
    def __init__(self, controller=None, task_queue=None, logger=None):
        self.router = APIRouter()
        self.controller = controller
        self.task_queue = task_queue
        self.logger = logger

        @self.router.get('/', tags=['Welcome'])
        async def welcome():
            try:
                return {
                    "message": "Welcome to WiSDON AI Training Platform !!",
                    "endpoints": {
                        "GET /fl_task_queue/show_waiting": "Show total tasks waiting in the queue",
                        "GET /fl_task_queue/show_current": "Show current training task",
                        "GET /fl_controller/show_status": "Show current flower controller status",
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Connection failed: {e}")
        
        # Show total tasks which waiting in the queue
        @self.router.get('/fl_task_queue/show_waiting', tags=['Show FL Retrain Event'])
        async def fl_task_queue_show_waiting():
            try:
                queue_content = self.task_queue.get_queue_content(self.task_queue.fl_task_queue)
                return {'queue_content': queue_content}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to show waiting retrain tasks: {e}")
        
        # Show current federated learning retrain task
        @self.router.get('/fl_task_queue/show_current', tags=['Show FL Retrain Event'])
        async def fl_task_queue_show_current():
            try:
                return {'current_retrain': self.task_queue.fl_current_training_task}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to show current retrain task: {e}")

        # Show current task's participants
        # @self.router.get('/fl_retrain_participants/show_current', tags=['Show FL Retrain Event'])
        # async def fl_participants_show_currnet():
        #     return {'current_participants' : self.controller.fl_retrain_participants}