#==========================================================
# Description: Define Ctrl plane API for controller, only called by AiTrPlat(server)
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/02/06
# Version: 0.1.0
# License: None
#==========================================================

from asyncio.log import logger
from logging import config
import sys, os
import requests
from typing import List, Optional, Dict
from fastapi import FastAPI, APIRouter, Response, status, HTTPException
from pydantic import BaseModel
import threading
from pymongo import MongoClient
import subprocess
import re

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
        self.thread = None
        self.stop_thread = threading.Event()
        
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
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to show controller status: {e}")
            
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
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to reset controller: {e}")
                
        
        # Raise flower controller task start event
        @self.router.post('/fl_controller/task_start', tags=['Flower Controller'])
        async def fl_controller_task_start():
            with threading.Lock():
                try:
                    self.controller.flower_controller.ai_tr_plat.raise_task_start()
                    # self.controller.flower_controller.run_cycle()
                    self.logger.info("Flower Controller raises task start event")
                    return {
                        "status": "success",
                        "message": "Flower Controller task start event raised"
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to raise task start event: {e}")

        def _run_flower_app(self, app_name):
            try:
                if app_name == "NES":
                    command = f"flwr run app/network-energy-saving network_energy_saving --stream"
                else:
                    command = f"flwr run app/{app_name} {app_name} --stream"
                self.logger.info("Launching Flower app in background: %s", command)

                exit_code = os.system(command)
                if exit_code == 0:
                    self.logger.info("Flower app completed successfully.")
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_training_complete()
                    # self.controller.flower_controller.run_cycle()
                else:
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_internal_error()
                    # self.controller.flower_controller.run_cycle()
                    self.logger.error("Flower app exited with code %s", exit_code)
            except OSError as exc:
                # Trigger the controller
                self.controller.flower_controller.ai_tr_plat.raise_internal_error()
                # self.controller.flower_controller.run_cycle()
                self.logger.error("Failed to execute Flower app: %s", exc)

        # Raise flower controller start flower app
        @self.router.post('/fl_controller/start_flower_app', tags=['Flower Controller'])
        async def fl_controller_start_flower_app():
            with threading.Lock():
                try:
                    self.task_queue.fl_training_queue_pop()
                    task_name = self.task_queue.fl_current_training_task
                    app_name = task_name["app_name"]

                    # Connect to MongoDB and save training parameters
                    mongodb_url = os.environ.get("AITRCOMMONDB_URI")
                    client = MongoClient(mongodb_url)
                    database = client["TrainingConfig"]
                    collection_name = f"{task_name['project_id']}_{task_name['app_name']}_{task_name['model_name']}_{task_name['model_version']}_{task_name['mode']}_{task_name['dataset_name']}"
                    collection = database[collection_name]

                    # Update data into MongoDB
                    filter = {
                        "project_id": task_name["project_id"],
                        "app_name": task_name["app_name"],
                        "model_name": task_name["model_name"],
                        "model_version": task_name["model_version"],
                        "mode": task_name["mode"],
                        "dataset_name": task_name["dataset_name"]
                    }
                    collection.update_one(filter, {"$set": {"status": "running"}}, upsert=False)

                    # Save current task name
                    collection_list = database.list_collection_names()
                    collection_name = "current_task"
                    if collection_name in collection_list:
                        database.drop_collection(collection_name)

                    collection = database[collection_name]
                    data = {
                        "project_id": task_name["project_id"],
                        "app_name": task_name["app_name"],
                        "model_name": task_name["model_name"],
                        "model_version": task_name["model_version"],
                        "mode": task_name["mode"],
                        "dataset_name": task_name["dataset_name"],
                        "status": "running"
                    }
                    collection.insert_one(data)

                    self.thread = threading.Thread(target=_run_flower_app, args=(self,app_name,), daemon=True)
                    self.thread.start()
                    self.logger.info("Flower app thread started.")
                except RuntimeError as exc:
                    self.logger.error("Could not start Flower app thread: %s", exc)
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_internal_error()
                    # self.controller.flower_controller.run_cycle()
                except Exception as e:
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_internal_error()
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to raise task start event: {e}")
                
        def get_current_run_id(self, app_name: str) -> Optional[str]:
            """
            執行 flwr run --list 並解析出最新的 RUN_ID。
            假設最新的 Run 會出現在列表的第一行或具有特定狀態。
            """
            try:

                # Change directory to where the flwr command is available if necessary
                if app_name == "NES":
                    work_dir = f"/app/app/network-energy-saving"
                else:
                    work_dir = f"/app/app/{app_name}"

                # 執行指令並取得輸出內容
                result = subprocess.run(
                    ["flwr", "list", "--runs"], 
                    cwd=work_dir,
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                
                output = result.stdout
                
                # 典型的 flwr list --runs 輸出格式通常包含表格
                # 我們尋找看起來像數字或特定 ID 格式的字串
                # 這裡假設 ID 是純數字，且我們取第一筆找到的資料
                lines = output.strip().split('\n')
                
                if len(lines) < 6:
                    self.logger.info("目前沒有偵測到任何運行中的 Run。")
                    return None

                # 簡單解析邏輯：跳過表頭，尋找第一行數據中的第一個欄位
                # 這裡使用正則表達式尋找每一行開頭的數字 (RUN_ID)
                for line in lines:
                    match = re.search(r'\d{10,}', line)
                    if match:
                        current_run_id = match.group(0)
                        return current_run_id
                        
                return None

            except subprocess.CalledProcessError as e:
                self.logger.info(f"執行指令失敗: {e}")
                return None
            except Exception as e:
                self.logger.info(f"發生錯誤: {e}")
                return None
    
        def _terminate_training(self, current_run_id, app_name):
            # command = "pkill -f 'flwr run'"
            # command = f"flwr stop {current_run_id}"

            if app_name == "NES":
                work_dir = f"/app/app/network-energy-saving"
            else:
                work_dir = f"/app/app/{app_name}"
            
            try:
                self.logger.info("Attempting to terminate Flower training process: %s", f"flwr stop {current_run_id}")
                # 執行指令並取得輸出內容
                result = subprocess.run(
                    ["flwr", "stop", f"{current_run_id}"], 
                    cwd=work_dir,
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                self.logger.info("Training process terminated successfully. %s", result.stdout)
                # Trigger the controller
                self.controller.flower_controller.ai_tr_plat.raise_termination_complete()
                self.logger.info("Flower Controller raises termination complete event")
            except OSError as exc:
                # Trigger the controller
                self.controller.flower_controller.ai_tr_plat.raise_termination_fail()
                # self.controller.flower_controller.run_cycle()
                self.logger.error("Failed to terminate training process: %s", result.stdout)

        # Raise flower controller terminate training event
        @self.router.post('/fl_controller/terminate_training', tags=['Flower Controller'])
        async def fl_controller_terminate_training():
            with threading.Lock():
                try:
                    # Connect to MongoDB and save training parameters
                    mongodb_url = os.environ.get("AITRCOMMONDB_URI")
                    client = MongoClient(mongodb_url)
                    database = client["TrainingConfig"]
                    collection_name = "current_task"
                    collection = database[collection_name]
                    filter = {
                        "status": "running"
                    }
                    item = collection.find_one(filter)
                    app_name = item["app_name"]

                    # Get current RUN_ID
                    current_run_id = get_current_run_id(self,app_name)
                    self.logger.info("Current RUN_ID for app %s: %s", app_name, current_run_id)
                    _terminate_training(self, current_run_id, app_name)
                    self.logger.info("Termination thread started.")
                except RuntimeError as exc:
                    self.logger.error("Could not start termination thread: %s", exc)
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_termination_fail()
                    # self.controller.flower_controller.run_cycle()
                except Exception as e:
                    # Trigger the controller
                    self.controller.flower_controller.ai_tr_plat.raise_termination_fail()
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to raise terminate training event: {e}")
                
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
                    # self.controller.flower_controller.run_cycle()
                    self.logger.info("Flower Controller raises create error log complete event")
                    return {
                        "status": "success",
                        "message": "Flower Controller create error log complete event raised"
                    }
                except Exception as e:
                    self.controller.flower_controller.ai_tr_plat.raise_create_error_log_complete()
                    # self.controller.flower_controller.run_cycle()
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to raise create error log event: {e}")
                

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
                    raise HTTPException(status_code=500, detail=f"[Error code 100: CONNECTION_ERROR] Error in API Server. Failed to raise send error log event: {e}")
                
        