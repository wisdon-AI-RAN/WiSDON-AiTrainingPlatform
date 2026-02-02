#==========================================================
# Description: Define Ctrl plane API for controller, only called by AiTrPlat(server)
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/01/20
# Version: 0.1.0
# License: None
#==========================================================

from asyncio.log import logger
import sys, os
import requests
from typing import List, Optional, Dict
from fastapi import FastAPI, APIRouter, Response, status, HTTPException
from pydantic import BaseModel
import threading
from pymongo import MongoClient

class ModelInfo(BaseModel):
    project_id: str
    app_name: str
    model_version: str

class StateResponse(BaseModel):
    """Model for state response"""
    flower_controller_current_state: str
    is_flower_controller_active: bool
    is_flower_controller_final: bool
    task_queue_controller_current_state: str
    is_task_queue_active: bool
    is_task_queue_final: bool
    
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
                try:
                    flower_controller_state_names = {
                        self.controller.flower_controller.State.flower_controller_s0__idle: "IDLE",
                        self.controller.flower_controller.State.flower_controller_s1__start_flower_app: "StartFlowerApp",
                        self.controller.flower_controller.State.flower_controller_s2__termination: "Termination",
                        self.controller.flower_controller.State.flower_controller_s3__internal_error: "InternalError",
                        self.controller.flower_controller.State.null_state: "NullState"
                    }
                    
                    current_state_id = self.controller.flower_controller._FlowerController__state_vector[0]
                    current_state_name = flower_controller_state_names.get(current_state_id, "Unknown")

                    task_queue_controller_state_names = {
                        self.controller.task_queue_controller.State.task_queue_s0_idel: "IDLE",
                        self.controller.task_queue_controller.State.task_queue_s1_queuing: "Queuing",
                        self.controller.task_queue_controller.State.task_queue_s2_waiting: "Waiting",
                        self.controller.task_queue_controller.State.task_queue_s3_complete: "Complete",
                        self.controller.task_queue_controller.State.null_state: "NullState"
                    }

                    current_state_id = self.controller.flower_controller._FlowerController__state_vector[0]
                    flower_controller_current_state_name = flower_controller_state_names.get(current_state_id, "Unknown")

                    current_state_id = self.controller.task_queue_controller._TaskQueue__state_vector[0]
                    task_queue_controller_current_state_name = task_queue_controller_state_names.get(current_state_id, "Unknown")
                    
                    return StateResponse(
                        flower_controller_current_state=flower_controller_current_state_name,
                        is_flower_controller_active=self.controller.flower_controller.is_active(),
                        is_flower_controller_final=self.controller.flower_controller.is_final(),
                        task_queue_controller_current_state=task_queue_controller_current_state_name,
                        is_task_queue_active=self.controller.task_queue_controller.is_active(),
                        is_task_queue_final=self.controller.task_queue_controller.is_final()
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to show controller status: {e}")
            
        # Reset controller
        @self.router.post('/fl_controller/reset', tags=['Flower Controller'])
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
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to reset controller: {e}")
                
        
        # Raise flower controller task start event
        @self.router.post('/fl_controller/task_start', tags=['Flower Controller'])
        async def fl_controller_task_start():
            with threading.Lock():
                try:
                    self.controller.flower_controller.ai_tr_plat.raise_task_start()
                    self.controller.flower_controller.run_cycle()
                    self.logger.info("Flower Controller raises task start event")
                    return {
                        "status": "success",
                        "message": "Flower Controller task start event raised"
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to raise task start event: {e}")

        def _run_flower_app(self, app_name):
            if app_name == "NES":
                command = f"flwr run app/network-energy-saving network_energy_saving --stream"
            else:
                command = f"flwr run app/{app_name} {app_name} --stream"
            logger.info("Launching Flower app in background: %s", command)
            try:
                exit_code = os.system(command)
                if exit_code == 0:
                    logger.info("Flower app completed successfully.")
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_training_complete()
                    self.controller.flower_controller.run_cycle()
                else:
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_internal_error()
                    self.controller.flower_controller.run_cycle()
                    logger.error("Flower app exited with code %s", exit_code)
            except OSError as exc:
                # Trigger the controller
                self.controller.flower_controller.ai_tr_plat.raise_internal_error()
                self.controller.flower_controller.run_cycle()
                logger.error("Failed to execute Flower app: %s", exc)

        # Raise flower controller start flower app
        @self.router.post('/fl_controller/start_flower_app', tags=['Flower Controller'])
        async def fl_controller_start_flower_app():
            with threading.Lock():
                try:
                    self.task_queue.fl_training_queue_pop()
                    task_name = self.task_queue.fl_current_training_task
                    app_name = task_name["app_name"]

                    # Connect to MongoDB and save training parameters
                    mongodb_url = os.environ.get("MONGODB_URL", "mongodb://mongodb:27017")
                    client = MongoClient(mongodb_url)
                    database = client["Training_Configuration"]
                    collection = database["training_conf"]

                    # Update data into MongoDB
                    filter = {
                        "project_id": task_name["project_id"],
                        "app_name": task_name["app_name"],
                        "model_version": task_name["model_version"],
                        "mode": task_name["mode"],
                    }
                    collection.update_one(filter, {"$set": {"status": "running"}}, upsert=False)

                    threading.Thread(target=_run_flower_app, args=(self,app_name,), daemon=True).start()
                    logger.info("Flower app thread started.")
                except RuntimeError as exc:
                    logger.error("Could not start Flower app thread: %s", exc)
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_internal_error()
                    self.controller.flower_controller.run_cycle()
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to raise task start event: {e}")
                
        # Raise flower controller stop training event
        @self.router.post('/fl_controller/task_stop', tags=['Flower Controller'])
        async def fl_controller_task_stop():
            with threading.Lock():
                try:
                    self.controller.flower_controller.ai_tr_plat.raise_stop_training()
                    self.controller.flower_controller.run_cycle()
                    self.logger.info("Flower Controller raises stop training event")
                    return {
                        "status": "success",
                        "message": "Flower Controller stop training event raised"
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to raise stop training event: {e}")
                
        def _terminate_training(self):
            command = "kill -f 'flwr run'"
            logger.info("Attempting to terminate Flower training process: %s", command)
            try:
                exit_code = os.system(command)
                if exit_code == 0:
                    logger.info("Training process terminated successfully.")
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_termination_complete()
                    self.controller.flower_controller.run_cycle()
                    logger.info("Flower Controller raises termination complete event")
                else:
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_termination_fail()
                    self.controller.flower_controller.run_cycle()
                    logger.warning("Terminate command exited with code %s", exit_code)
            except OSError as exc:
                # Trigger the controller
                self.controller.flower_controller.ai_tr_plat.raise_termination_fail()
                self.controller.flower_controller.run_cycle()
                logger.error("Failed to terminate training process: %s", exc)

        # Raise flower controller terminate training event
        @self.router.post('/fl_controller/terminate_training', tags=['Flower Controller'])
        async def fl_controller_terminate_training():
            with threading.Lock():
                try:
                    threading.Thread(target=_terminate_training, daemon=True).start()
                    logger.info("Termination thread started.")
                except RuntimeError as exc:
                    logger.error("Could not start termination thread: %s", exc)
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_termination_fail()
                    self.controller.flower_controller.run_cycle()
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to raise terminate training event: {e}")
                
        # Raise flower controller create error log event
        @self.router.post('/fl_controller/create_error_log', tags=['Flower Controller'])
        async def fl_controller_create_error_loge():
            with threading.Lock():
                try:
                    #===== Create error log event =====#
                    #
                    #
                    #
                    #
                    #

                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_create_error_log_complete()
                    self.controller.flower_controller.run_cycle()
                    self.logger.info("Flower Controller raises create error log complete event")
                    return {
                        "status": "success",
                        "message": "Flower Controller create error log complete event raised"
                    }
                except Exception as e:
                    self.controller.flower_controller.AiTrPlat.raise_create_error_log_complete()
                    self.controller.flower_controller.run_cycle()
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to raise create error log event: {e}")
                

        # Flower controller send error log to AppPlat
        @self.router.post('/fl_controller/send_error_log', tags=['Flower Controller'])
        async def fl_controller_send_error_log():
            with threading.Lock():
                try:
                    #===== Send error log event =====#
                    #
                    #
                    #
                    #
                    #

                    # Trigger the controller
                    self.logger.info("Flower Controller raises send error log complete event")
                    return {
                        "status": "success",
                        "message": "Flower Controller send error log complete event raised"
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server-Failed to raise send error log event: {e}")
                
        