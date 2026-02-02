import sys, os
import requests

class CallBackFunc():
    def get_app_status():
        try:
            print("Get App Status")
        except Exception as e:
            print("Error Code 202 [IN_EVENT_ERROR]: Error in Task Queue Controller in event get_app_status:", e)