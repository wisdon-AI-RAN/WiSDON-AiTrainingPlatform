import sys, os
import requests

class CallBackFunc():
    def get_app_status():
        api_server_url = os.environ.get("API_SERVER_URL", "http://localhost:9005")
        controller_url = f"{api_server_url}/fl_training_task_queue/is_delete_running"
        try:
            response = requests.post(controller_url, timeout=10)
            print("Get App Status")
            return response.json()["is_delete_running"]
        except Exception as e:
            print("Error Code 202 [IN_EVENT_ERROR]: Error in Task Queue Controller in event get_app_status:", e)