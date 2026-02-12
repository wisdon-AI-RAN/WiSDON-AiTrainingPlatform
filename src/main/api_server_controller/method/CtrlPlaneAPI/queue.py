#==========================================================
# Description: Define Ctrl plane API for task queue, only called by AiTrPlat(server). 
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/01/20
# Version: 0.1.0
# License: None
#==========================================================

import sys, os
from typing import List, Optional, Dict

import requests
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
            try:
                # Trigger the controller 
                self.task_queue.fl_training_queue_push(self.controller.task_name, "0")
                # self.controller.task_queue_controller.run_cycle()
                #===== need to report to AppPlat here =====#
                # push success
                #
                #
                #
                #=========================================#
            except Exception as e:
                # self.controller.task_queue_controller.run_cycle()
                raise HTTPException(status_code=500, detail=f"[Error Code 100: CONNECTION_ERROR]: Failed to push task: {e}")
            
        # Called by AiTrPlat to delete task from server waiting queue
        @self.router.post('/fl_training_task_queue/delete', tags=['Control Task Queue'])
        async def fl_training_task_delete():
            try:
                # Trigger the controller 
                self.task_queue.fl_training_queue_delete(self.controller.delete_name)
                self.controller.flower_controller.ai_tr_plat.raise_stop_training()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error Code 100: CONNECTION_ERROR]: Failed to delete task: {e}")
            
        # Called by AiTrPlat to inform task is completed
        @self.router.post('/fl_training_task_queue/complete', tags=['Control Task Queue'])
        async def fl_training_task_complete():
            try:
                #===== need to report to AppPlat here =====#
                # run success
                #
                #
                #
                #=========================================#
                # Trigger the controller 
                self.task_queue.fl_training_queue_task_finish()
                self.controller.task_queue_controller.ai_tr_plat.raise_task_complete()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error Code 100: CONNECTION_ERROR]: Failed to complete task: {e}")
            
        # Called by AiTrPlat to report push task fail
        @self.router.post('/fl_training_task_queue/report_push_fail', tags=['Control Task Queue'])
        async def fl_training_task_report_push_fail():
            try:
                pass
                #===== need to report to AppPlat here =====#
                # push fail
                #
                #
                #
                #=========================================#
                # self.controller.task_queue_controller.run_cycle()
            except Exception as e:
                # self.controller.task_queue_controller.run_cycle()
                raise HTTPException(status_code=500, detail=f"[Error Code 100: CONNECTION_ERROR]: Failed to run task queue controller cycle: {e}")
            
        # Called by AiTrPlat to get delete task is running 
        @self.router.get('/fl_training_task_queue/is_delete_running', tags=['Control Task Queue'])
        async def fl_training_task_is_delete_running():
            try:
                delete_task_name = self.controller.delete_name
                if self.task_queue.fl_current_training_task == delete_task_name:
                    return {"is_delete_running": True}
                else:
                    return {"is_delete_running": False}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"[Error Code 100: CONNECTION_ERROR]: Failed to get delete task running status: {e}")