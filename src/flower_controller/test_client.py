"""
Test client for the Flower Controller API
This script demonstrates how to interact with the controller via FastAPI
"""

import requests
import time
import json

# Base URL of the API
BASE_URL = "http://localhost:9005"

def print_response(response, action):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{action}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def get_state():
    """Get current state"""
    response = requests.get(f"{BASE_URL}/state")
    print_response(response, "GET STATE")
    return response.json()

def trigger_event(event_name):
    """Trigger an event"""
    response = requests.post(
        f"{BASE_URL}/event",
        json={"event_name": event_name}
    )
    print_response(response, f"TRIGGER EVENT: {event_name}")
    return response.json()

def run_cycle():
    """Run a cycle"""
    response = requests.post(f"{BASE_URL}/run_cycle")
    print_response(response, "RUN CYCLE")
    return response.json()

def trigger_and_run(event_name):
    """Trigger event and run cycle in one call"""
    response = requests.post(
        f"{BASE_URL}/trigger_and_run",
        json={"event_name": event_name}
    )
    print_response(response, f"TRIGGER AND RUN: {event_name}")
    return response.json()

def get_available_events():
    """Get available events"""
    response = requests.get(f"{BASE_URL}/available_events")
    print_response(response, "AVAILABLE EVENTS")
    return response.json()

def reset_controller():
    """Reset controller"""
    response = requests.post(f"{BASE_URL}/reset")
    print_response(response, "RESET CONTROLLER")
    return response.json()

def main():
    """Run a test scenario"""
    print("\n" + "="*60)
    print("FLOWER CONTROLLER API TEST CLIENT")
    print("="*60)
    
    try:
        # Test 1: Get initial state
        print("\n\n1. Getting initial state...")
        get_state()
        time.sleep(0.5)
        
        # Test 2: Get available events
        print("\n\n2. Getting available events...")
        get_available_events()
        time.sleep(0.5)
        
        # Test 3: Start a task (IDLE -> StartFlowerApp)
        print("\n\n3. Starting a task (IDLE -> StartFlowerApp)...")
        trigger_and_run("task_start")
        time.sleep(0.5)
        
        # Test 4: Check state
        print("\n\n4. Checking state after task start...")
        state = get_state()
        assert state['current_state'] == "StartFlowerApp", "State should be StartFlowerApp"
        time.sleep(0.5)
        
        # Test 5: Complete training (StartFlowerApp -> IDLE)
        print("\n\n5. Completing training (StartFlowerApp -> IDLE)...")
        trigger_and_run("training_complete")
        time.sleep(0.5)
        
        # Test 6: Check state
        print("\n\n6. Checking state after training complete...")
        state = get_state()
        assert state['current_state'] == "IDLE", "State should be IDLE"
        time.sleep(0.5)
        
        # Test 7: Start task again and then stop it
        print("\n\n7. Starting task again...")
        trigger_and_run("task_start")
        time.sleep(0.5)
        
        print("\n\n8. Stopping training (StartFlowerApp -> Termination)...")
        trigger_and_run("stop_training")
        time.sleep(0.5)
        
        # Test 8: Check state
        print("\n\n9. Checking state after stop...")
        state = get_state()
        assert state['current_state'] == "Termination", "State should be Termination"
        time.sleep(0.5)
        
        # Test 9: Complete termination
        print("\n\n10. Completing termination (Termination -> IDLE)...")
        trigger_and_run("termination_complete")
        time.sleep(0.5)
        
        # Test 10: Reset controller
        print("\n\n11. Resetting controller...")
        reset_controller()
        time.sleep(0.5)
        
        # Test 11: Final state check
        print("\n\n12. Final state check...")
        state = get_state()
        assert state['current_state'] == "IDLE", "State should be IDLE after reset"
        
        print("\n\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n\nERROR: Could not connect to the API server.")
        print("Make sure the server is running with: python controller_main.py")
    except AssertionError as e:
        print(f"\n\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\n\nERROR: {e}")

if __name__ == "__main__":
    main()
