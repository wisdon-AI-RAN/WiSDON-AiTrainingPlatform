import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Adjust path to ensure we can import the local modules even if they shadow stdlib
# This adds the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Attempting to import Platform and TaskQueue
# In case of naming conflicts (like 'platform'), proper package structure or running as module is recommended.
# Assuming this script is run from its directory or properly configured pythonpath.

try:
    # Try importing as if they are local packages
    from platform.srcgen.platform import Platform
except ImportError:
    # Fallback or specific handling if needed. 
    # Note: 'platform' is a standard library module. 
    # If this fails, consider renaming the folder or running as a package from higher level.
    # For now, we assume the environment is set up to allow this import or we use a workaround.
    import importlib.util
    spec = importlib.util.spec_from_file_location("PlatformModule", os.path.join(os.path.dirname(__file__), "platform/srcgen/platform.py"))
    PlatformModule = importlib.util.module_from_spec(spec)
    sys.modules["PlatformModule"] = PlatformModule
    spec.loader.exec_module(PlatformModule)
    Platform = PlatformModule.Platform

try:
    from task_queue.srcgen.task_queue import TaskQueue
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("TaskQueueModule", os.path.join(os.path.dirname(__file__), "task_queue/srcgen/task_queue.py"))
    TaskQueueModule = importlib.util.module_from_spec(spec)
    sys.modules["TaskQueueModule"] = TaskQueueModule
    spec.loader.exec_module(TaskQueueModule)
    TaskQueue = TaskQueueModule.TaskQueue

app = FastAPI()

# Observer implementation to bridge events
class FunctionObserver:
    def __init__(self, callback):
        self.callback = callback
    
    def next(self, value=None):
        self.callback(value)

class Controller:
    def __init__(self):
        self.platform = Platform()
        self.task_queue = TaskQueue()
        self._setup()
    
    def _setup(self):
        # Setup operation callbacks (Mocking required interfaces)
        class PlatOps:
            def get_num_data(self):
                # Retrieve actual data count if possible, or mock
                return 100 
        
        class TaskOps:
            def get_app_status(self):
                # Retrieve app status if possible, or mock
                return True 

        # Set callbacks
        self.platform.ai_tr_plat.operation_callback = PlatOps()
        self.task_queue.ai_tr_plat.operation_callback = TaskOps()
        
        # --- Connect Platform -> TaskQueue ---
        
        # Platform: push_in_task_queue -> TaskQueue: raise_push_in_task_queue
        self.platform.ai_tr_plat.push_in_task_queue_observable.subscribe(
            FunctionObserver(self._on_platform_push_in_task_queue)
        )
        
        # Platform: delete_app -> TaskQueue: raise_delete_app
        self.platform.ai_tr_plat.delete_app_observable.subscribe(
             FunctionObserver(self._on_platform_delete_app)
        )

        # --- Connect TaskQueue -> Platform ---
        
        # TaskQueue: push_task_complete -> Platform: raise_push_task_complete
        self.task_queue.ai_tr_plat.push_task_complete_observable.subscribe(
             FunctionObserver(self._on_tq_push_task_complete)
        )

        # TaskQueue: push_task_fail -> Platform: raise_push_task_fail
        self.task_queue.ai_tr_plat.push_task_fail_observable.subscribe(
             FunctionObserver(self._on_tq_push_task_fail)
        )
        
        # Initial activation
        self.platform.enter()
        self.task_queue.enter()
        
    def _on_platform_push_in_task_queue(self, value=None):
        print("Event: Platform request push_in_task_queue")
        self.task_queue.ai_tr_plat.raise_push_in_task_queue()
        self.task_queue.run_cycle()

    def _on_platform_delete_app(self, value=None):
        print("Event: Platform request delete_app")
        self.task_queue.ai_tr_plat.raise_delete_app()
        self.task_queue.run_cycle()

    def _on_tq_push_task_complete(self, value=None):
        print("Event: TaskQueue reported push_task_complete")
        self.platform.ai_tr_plat.raise_push_task_complete()
        self.platform.run_cycle()

    def _on_tq_push_task_fail(self, value=None):
        print("Event: TaskQueue reported push_task_fail")
        self.platform.ai_tr_plat.raise_push_task_fail()
        self.platform.run_cycle()

    # Public Triggers
    def trigger_retrain(self):
        self.platform.app_plat.raise_retrain()
        self.platform.run_cycle()

    def trigger_delete(self):
        self.platform.app_plat.raise_delete()
        self.platform.run_cycle()

    def trigger_receive_report(self):
        self.platform.app_plat.raise_receive_report()
        self.platform.run_cycle()
        
    def trigger_task_complete(self):
        # Simulate a task processing completion in the queue
        self.task_queue.ai_tr_plat.raise_task_complete()
        self.task_queue.run_cycle()

# Instantiate controller
controller = Controller()

@app.post("/platform/retrain")
async def retrain():
    """Trigger the retrain event on the Platform."""
    controller.trigger_retrain()
    return {"message": "Retrain event triggered"}

@app.post("/platform/delete")
async def delete_app():
    """Trigger the delete event on the Platform."""
    controller.trigger_delete()
    return {"message": "Delete event triggered"}

@app.post("/platform/report")
async def receive_report():
    """Trigger the receive_report event on the Platform."""
    controller.trigger_receive_report()
    return {"message": "Receive report event triggered"}

@app.post("/task_queue/task_complete")
async def task_complete():
    """Trigger the task_complete event on the TaskQueue."""
    controller.trigger_task_complete()
    return {"message": "Task complete event triggered"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
