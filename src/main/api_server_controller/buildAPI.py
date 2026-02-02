#==========================================================
# Description: Build API Server for AI Training Platform Ctrl Plane/ Data Plane API.
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/01/20
# Version: 0.1.0
# License: None
#==========================================================

import logging
from typing import List, Optional, Dict
from fastapi import FastAPI
from contextlib import asynccontextmanager
from controller.controller_main import Controller
from training_task_queue import TrainingTaskQueue
from method import *

# Create logger
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] uvicorn - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize controller
main_controller = Controller(logger=logger)
main_task_queue = TrainingTaskQueue(logger=logger)

# Create events with start up and shutdown for api server
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("lifespan init !")
#     processing_thread = threading.Thread(target=task_queue.fl_training_queue_pop, daemon=True)
#     processing_thread.start()
#     try:
#         yield
#     finally:
#         logger.info("lifespan close !")

# Initialize FastAPI application
# app = FastAPI(title='WiSDON AI Training Platform - API Server', lifespan=lifespan)
app = FastAPI(title='WiSDON AI Training Platform - API Server')

"""
    Define api functions
"""
#===== Initialize Ctrl Plane API =====#
general_ctrl_api = GeneralCtrlAPI(controller=main_controller, task_queue=main_task_queue, logger=logger)
training_api = TrainingAPI(controller=main_controller, task_queue=main_task_queue, logger=logger)
controller_api = ControllerAPI(controller=main_controller, task_queue=main_task_queue, logger=logger)
task_queue_api = TaskQueueAPI(controller=main_controller, task_queue=main_task_queue, logger=logger)

#===== Initialize Data Plane API =====#
# model_conf_api = model_conf_API(controller=controller,
#                                 db_name = "Model_Configuration",
#                                 collection_name = "model_conf",
#                                 mongodb_uri =  "mongodb://localhost:27017/")
# model_location_api = model_pull_API(controller=controller,
#                                     db_name = "Model_Location", 
#                                     collection_name = "model_location",
#                                     mongodb_uri = "mongodb://localhost:27017/")
# ue_info_api = UE_info_API(db_name = "Test_Database",
#                         collection_name = "Test_Collection",
#                         mongodb_uri = "mongodb://localhost:27017/",
#                         batch_size = 1)
# record_update_api = record_update_API(mongodb_uri = "mongodb://localhost:27017/")

#===== Include routers =====#
app.include_router(general_ctrl_api.router)
app.include_router(training_api.router)
app.include_router(controller_api.router)
app.include_router(task_queue_api.router)

# app.include_router(model_conf_api.router)
# app.include_router(model_location_api.router)
# app.include_router(ue_info_api.router)
# app.include_router(record_update_api.router)
