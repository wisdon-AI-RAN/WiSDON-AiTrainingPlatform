#==========================================================
# Description: Define Ctrl plane API for controller, only called by AiTrPlat(server)
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

class StateResponse(BaseModel):
    """Model for state response"""
    current_state: str
    is_active: bool
    is_final: bool
    
class ControllerAPI:
    """
    Define Ctrl plane API for controller, only called by AiTrPlat(server).
    """
    def __init__(self, controller=None, task_queue=None, logger=None):
        self.router = APIRouter()
        self.controller = controller
        self.task_queue = task_queue
        self.logger = logger
        
        ''''''''' Flower Controller '''''''''
        # Show current flower controller status
        @self.router.get('/fl_controller/show_status', tags=['Flower Controller'])
        async def fl_controller_show_status():
            """Get the current state of the controller"""
            with threading.Lock():
                state_names = {
                    self.controller.flower_controller.State.flower_controller_s0__idle: "IDLE",
                    self.controller.flower_controller.State.flower_controller_s1__start_flower_app: "StartFlowerApp",
                    self.controller.flower_controller.State.flower_controller_s2__termination: "Termination",
                    self.controller.flower_controller.State.flower_controller_s3__internal_error: "InternalError",
                    self.controller.flower_controller.State.null_state: "NullState"
                }
                
                current_state_id = self.controller.flower_controller._FlowerController__state_vector[0]
                current_state_name = state_names.get(current_state_id, "Unknown")
                
                return StateResponse(
                    current_state=current_state_name,
                    is_active=self.controller.flower_controller.is_active(),
                    is_final=self.controller.flower_controller.is_final()
                )
            
        # Reset controller
        @self.router.get('/fl_controller/reset', tags=['Flower Controller'])
        async def fl_controller_reset():
            """Reset the controller to initial state"""
            with threading.Lock():
                try:
                    self.controller.flower_controller.exit()
                    self.controller.flower_controller.enter()
                    self.logger.info("Flower Controller reset to initial state")

                    self.controller.task_queue_controller.exit()
                    self.controller.task_queue_controller.enter()
                    self.logger.info("Task Queue Controller reset to initial state")

                    return {
                        "status": "success",
                        "message": "Controller reset to IDLE state"
                    }
                except Exception as e:
                    self.logger.error(f"Error resetting controller: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
        