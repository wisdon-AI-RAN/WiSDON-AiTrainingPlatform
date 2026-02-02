# Flower Controller FastAPI Implementation

A FastAPI-based controller built on top of FlowerController state machine, allowing HTTP-based state transitions and command triggering.

## Features

- **REST API** for controller state management
- **State transitions** triggered via HTTP endpoints
- **Real-time state monitoring**
- **Command output** via observable subscriptions
- **Thread-safe** operations

## State Machine Overview

The controller has the following states:

- **IDLE**: Initial state, waiting for tasks
- **StartFlowerApp**: Training is running
- **Termination**: Training is being terminated
- **InternalError**: Error state

## Available Events

| Event | Description | Valid From State |
|-------|-------------|------------------|
| `task_start` | Start a new task | IDLE |
| `training_complete` | Training completed successfully | StartFlowerApp |
| `stop_training` | Stop training | StartFlowerApp |
| `termination_complete` | Termination successful | Termination |
| `termination_fail` | Termination failed | Termination |
| `internal_error` | Internal error occurred | StartFlowerApp |
| `create_error_log_complete` | Error log created | InternalError |

## Installation

### Option 1: Docker (Recommended)

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

Or build and run manually:
```bash
docker build -t flower-controller .
docker run -d -p 9000:9000 --name flower-controller flower-controller
```

### Option 2: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

**With Docker:**
```bash
docker-compose up -d
```

**Without Docker:**
```bash
python controller_main.py
```

The server will start on `http://localhost:9000`

### API Endpoints

#### 1. Get Current State
```bash
curl http://localhost:9000/state
```

Response:
```json
{
  "current_state": "IDLE",
  "is_active": true,
  "is_final": false
}
```

#### 2. List Available Events
```bash
curl http://localhost:9000/available_events
```

#### 3. Trigger an Event
```bash
curl -X POST http://localhost:9000/event \
  -H "Content-Type: application/json" \
  -d '{"event_name": "task_start"}'
```

#### 4. Run State Machine Cycle
```bash
curl -X POST http://localhost:9000/run_cycle
```

#### 5. Trigger Event and Run (Convenience Endpoint)
```bash
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "task_start"}'
```

#### 6. Reset Controller
```bash
curl -X POST http://localhost:9000/reset
```

### Interactive API Documentation

Visit `http://localhost:9000/docs` for interactive Swagger UI documentation.

## Testing

Run the test client to see a complete workflow:

```bash
python test_client.py
```

## Example Workflow

### Scenario 1: Successful Training
```bash
# 1. Start from IDLE
curl http://localhost:9000/state

# 2. Start task (IDLE -> StartFlowerApp)
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "task_start"}'

# 3. Complete training (StartFlowerApp -> IDLE)
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "training_complete"}'
```

### Scenario 2: Stopped Training
```bash
# 1. Start task
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "task_start"}'

# 2. Stop training (StartFlowerApp -> Termination)
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "stop_training"}'

# 3. Complete termination (Termination -> IDLE)
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "termination_complete"}'
```

### Scenario 3: Error Handling
```bash
# 1. Start task
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "task_start"}'

# 2. Trigger error (StartFlowerApp -> InternalError)
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "internal_error"}'

# 3. Complete error log (InternalError -> IDLE)
curl -X POST http://localhost:9000/trigger_and_run \
  -H "Content-Type: application/json" \
  -d '{"event_name": "create_error_log_complete"}'
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:9000"

# Get current state
response = requests.get(f"{BASE_URL}/state")
print(response.json())

# Start a task
response = requests.post(
    f"{BASE_URL}/trigger_and_run",
    json={"event_name": "task_start"}
)
print(response.json())

# Complete training
response = requests.post(
    f"{BASE_URL}/trigger_and_run",
    json={"event_name": "training_complete"}
)
print(response.json())
```

## Architecture

- **FastAPI**: Web framework for REST API
- **FlowerController**: State machine implementation
- **Observables**: RxPY-based event subscriptions for commands
- **Threading**: Thread-safe state machine access

## Observable Subscriptions

The controller subscribes to these observables for command outputs:

- `start_flower_app_observable`: Triggered when starting Flower app
- `terminate_training_observable`: Triggered when terminating training
- `create_error_log_observable`: Triggered when creating error log
- `task_complete_observable`: Triggered when task completes
- `task_error_observable`: Triggered on task error

These are logged and can be extended to trigger actual commands.

## Extending the Controller

To add custom command handling, modify the observable subscriptions in the `startup_event` function:

```python
controller.ai_tr_plat.start_flower_app_observable.subscribe(
    lambda: your_custom_function()
)
```

## Notes

- The controller uses a lock for thread-safe operations
- Events must be followed by `run_cycle()` to process them
- Use `trigger_and_run` endpoint for convenience
- State transitions are validated by the state machine

## License

See the main project LICENSE file.
