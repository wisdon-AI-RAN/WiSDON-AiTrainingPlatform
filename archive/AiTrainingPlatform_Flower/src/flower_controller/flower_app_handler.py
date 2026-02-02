"""
Flower App Handler - Implements observers to control the Flower application
"""
import subprocess
import logging
import os
import threading
import signal
from src.yakindu.rx import (
    StartFlowerAppObserver,
    TerminateTrainingObserver,
    CreateErrorLogObservableObserver,
    TaskCompleteObserver,
    TaskErrorObserver
)

logger = logging.getLogger(__name__)


class FlowerAppHandler:
    """Handler for controlling Flower application lifecycle"""
    
    def __init__(self, app_path="/app/network-energy-saving", container_name="superexec-serverapp"):
        """
        Initialize the Flower App Handler
        
        Args:
            app_path: Path to the Flower app directory
            container_name: Name of the Docker container running the app
        """
        self.app_path = app_path
        self.container_name = container_name
        self.process = None
        self.process_lock = threading.Lock()
        
    def start_flower_app(self):
        """Start the Flower application"""
        try:
            logger.info("Starting Flower application...")
            
            # Check if already running
            with self.process_lock:
                if self.process and self.process.poll() is None:
                    logger.warning("Flower app is already running")
                    return True
                
                # Run the install script to start the app
                install_script = os.path.join(self.app_path, "install_app.sh")
                
                if os.path.exists(install_script):
                    self.process = subprocess.Popen(
                        ["bash", install_script],
                        cwd=self.app_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid
                    )
                    logger.info(f"Flower app started with PID: {self.process.pid}")
                    return True
                else:
                    logger.error(f"Install script not found: {install_script}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error starting Flower app: {e}", exc_info=True)
            return False
    
    def terminate_training(self):
        """Terminate the training process"""
        try:
            logger.info("Terminating Flower training...")
            
            with self.process_lock:
                # Try to stop Docker container
                try:
                    result = subprocess.run(
                        ["docker", "stop", self.container_name],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        logger.info(f"Docker container {self.container_name} stopped successfully")
                    else:
                        logger.warning(f"Failed to stop container: {result.stderr}")
                except subprocess.TimeoutExpired:
                    logger.warning("Timeout while stopping container, forcing kill...")
                    subprocess.run(["docker", "kill", self.container_name], capture_output=True)
                
                # Terminate local process if exists
                if self.process and self.process.poll() is None:
                    try:
                        # Send SIGTERM to process group
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                        self.process.wait(timeout=10)
                        logger.info("Flower app process terminated")
                    except subprocess.TimeoutExpired:
                        # Force kill if doesn't respond
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        logger.warning("Flower app process force killed")
                    except Exception as e:
                        logger.error(f"Error terminating process: {e}")
                    finally:
                        self.process = None
                
            return True
            
        except Exception as e:
            logger.error(f"Error terminating training: {e}", exc_info=True)
            return False
    
    def create_error_log(self, error_message=None):
        """Create error log"""
        try:
            logger.info("Creating error log...")
            
            log_dir = "/app/logs"
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "flower_controller_error.log")
            
            with open(log_file, "a") as f:
                from datetime import datetime
                timestamp = datetime.now().isoformat()
                f.write(f"\n[{timestamp}] ERROR: {error_message or 'Unknown error'}\n")
                
                # Capture process error output if available
                with self.process_lock:
                    if self.process:
                        _, stderr = self.process.communicate(timeout=5)
                        if stderr:
                            f.write(f"Process stderr: {stderr.decode('utf-8', errors='ignore')}\n")
            
            logger.info(f"Error log created at: {log_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating error log: {e}", exc_info=True)
            return False


class StartFlowerAppObserverImpl(StartFlowerAppObserver):
    """Observer implementation for starting Flower app"""
    
    def __init__(self, handler, controller):
        self.handler = handler
        self.controller = controller
    
    def next(self, value=None):
        """Called when start_flower_app event is triggered"""
        logger.info("StartFlowerApp event triggered")
        
        success = self.handler.start_flower_app()
        
        if not success:
            # If failed, raise internal error
            logger.error("Failed to start Flower app, raising internal error")
            self.controller.ai_tr_plat.raise_internal_error()
            self.controller.run_cycle()


class TerminateTrainingObserverImpl(TerminateTrainingObserver):
    """Observer implementation for terminating training"""
    
    def __init__(self, handler, controller):
        self.handler = handler
        self.controller = controller
    
    def next(self, value=None):
        """Called when terminate_training event is triggered"""
        logger.info("TerminateTraining event triggered")
        
        success = self.handler.terminate_training()
        
        if success:
            # Signal termination complete
            self.controller.ai_tr_plat.raise_termination_complete()
        else:
            # Signal termination fail
            self.controller.ai_tr_plat.raise_termination_fail()
        
        self.controller.run_cycle()


class CreateErrorLogObserverImpl(CreateErrorLogObservableObserver):
    """Observer implementation for creating error log"""
    
    def __init__(self, handler, controller):
        self.handler = handler
        self.controller = controller
    
    def next(self, value=None):
        """Called when create_error_log event is triggered"""
        logger.info("CreateErrorLog event triggered")
        
        self.handler.create_error_log(error_message=value)
        
        # Signal error log complete
        self.controller.ai_tr_plat.raise_create_error_log_complete()
        self.controller.run_cycle()


class TaskCompleteObserverImpl(TaskCompleteObserver):
    """Observer implementation for task completion"""
    
    def __init__(self):
        pass
    
    def next(self, value=None):
        """Called when task_complete event is triggered"""
        logger.info("TaskComplete event triggered - Task finished successfully")


class TaskErrorObserverImpl(TaskErrorObserver):
    """Observer implementation for task error"""
    
    def __init__(self):
        pass
    
    def next(self, value=None):
        """Called when task_error event is triggered"""
        logger.error("TaskError event triggered - Task completed with errors")
