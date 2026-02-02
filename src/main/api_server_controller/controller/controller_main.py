import sys
import os
import threading

# Add the current directory to sys.path to ensure we can import the sub-controllers
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

from controller.flower.flower_controller import FlowerController
from controller.task_queue.task_queue import TaskQueue

class Controller:
    def __init__(self, logger=None):
        self.logger = logger
        self.flower_controller = FlowerController()
        self.task_queue_controller = TaskQueue()
        self.task_name = None
        self.delete_name = None

        """Initialize the controller when the API starts"""
        with threading.Lock():
            self.flower_controller.enter()
            self.logger.info("FlowerController initialized and entered")

            self.task_queue_controller.enter()
            self.logger.info("Task Queue Controller initialized and entered")