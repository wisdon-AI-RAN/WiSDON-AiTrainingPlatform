from srcgen.flower_controller import FlowerController
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Flower Controller API", version="1.0.0")

# Initialize the FlowerController
controller = FlowerController()

# Pydantic models for API requests
class EventRequest(BaseModel):
    """Model for triggering events"""
    event_name: str

class CommandRequest(BaseModel):
    """Model for sending commands"""
    command: str
    parameters: Optional[dict] = None

class StateResponse(BaseModel):
    """Model for state response"""
    current_state: str
    is_active: bool
    is_final: bool

# Global lock for thread safety
controller_lock = threading.Lock()

# Initialize controller on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the controller when the API starts"""
    with controller_lock:
        controller.enter()
        logger.info("FlowerController initialized and entered")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Flower Controller API",
        "endpoints": {
            "GET /state": "Get current state",
            "POST /event": "Trigger an event",
            "POST /run_cycle": "Run one cycle of the state machine",
            "GET /available_events": "List available events"
        }
    }

@app.get("/state", response_model=StateResponse)
async def get_state():
    """Get the current state of the controller"""
    with controller_lock:
        state_names = {
            FlowerController.State.flower_controller_s0__idle: "IDLE",
            FlowerController.State.flower_controller_s1__start_flower_app: "StartFlowerApp",
            FlowerController.State.flower_controller_s2__termination: "Termination",
            FlowerController.State.flower_controller_s3__internal_error: "InternalError",
            FlowerController.State.null_state: "NullState"
        }
        
        current_state_id = controller._FlowerController__state_vector[0]
        current_state_name = state_names.get(current_state_id, "Unknown")
        
        return StateResponse(
            current_state=current_state_name,
            is_active=controller.is_active(),
            is_final=controller.is_final()
        )

@app.get("/available_events", response_model=dict)
async def get_available_events():
    """Get list of available events that can be triggered"""
    return {
        "events": [
            "task_start",
            "training_complete",
            "stop_training",
            "termination_complete",
            "termination_fail",
            "internal_error",
            "create_error_log_complete"
        ],
        "description": {
            "task_start": "Start a new task (from IDLE state)",
            "training_complete": "Signal training completion (from StartFlowerApp state)",
            "stop_training": "Stop the training (from StartFlowerApp state)",
            "termination_complete": "Termination completed successfully (from Termination state)",
            "termination_fail": "Termination failed (from Termination state)",
            "internal_error": "Internal error occurred (from StartFlowerApp state)",
            "create_error_log_complete": "Error log created (from InternalError state)"
        }
    }

@app.post("/event", response_model=dict)
async def trigger_event(request: EventRequest):
    """Trigger an event in the state machine"""
    with controller_lock:
        event_name = request.event_name
        
        try:
            # Get the raise method for the event
            if hasattr(controller.ai_tr_plat, f"raise_{event_name}"):
                raise_method = getattr(controller.ai_tr_plat, f"raise_{event_name}")
                raise_method()
                logger.info(f"Event '{event_name}' raised")
                
                return {
                    "status": "success",
                    "message": f"Event '{event_name}' triggered",
                    "note": "Call /run_cycle to process the event"
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown event: {event_name}"
                )
        except Exception as e:
            logger.error(f"Error triggering event: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_cycle", response_model=StateResponse)
async def run_cycle():
    """Run one cycle of the state machine to process events"""
    with controller_lock:
        try:
            controller.run_cycle()
            logger.info("State machine cycle executed")
            
            # Get updated state
            state_names = {
                FlowerController.State.flower_controller_s0__idle: "IDLE",
                FlowerController.State.flower_controller_s1__start_flower_app: "StartFlowerApp",
                FlowerController.State.flower_controller_s2__termination: "Termination",
                FlowerController.State.flower_controller_s3__internal_error: "InternalError",
                FlowerController.State.null_state: "NullState"
            }
            
            current_state_id = controller._FlowerController__state_vector[0]
            current_state_name = state_names.get(current_state_id, "Unknown")
            
            return StateResponse(
                current_state=current_state_name,
                is_active=controller.is_active(),
                is_final=controller.is_final()
            )
        except Exception as e:
            logger.error(f"Error running cycle: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger_and_run", response_model=StateResponse)
async def trigger_and_run(request: EventRequest):
    """Trigger an event and immediately run a cycle (convenience endpoint)"""
    await trigger_event(request)
    return await run_cycle()

@app.post("/reset", response_model=dict)
async def reset_controller():
    """Reset the controller to initial state"""
    with controller_lock:
        try:
            controller.exit()
            controller.enter()
            logger.info("Controller reset to initial state")
            return {
                "status": "success",
                "message": "Controller reset to IDLE state"
            }
        except Exception as e:
            logger.error(f"Error resetting controller: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")

