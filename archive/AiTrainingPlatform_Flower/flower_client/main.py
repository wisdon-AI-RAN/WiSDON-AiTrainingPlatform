# from gRPCSymptomInfoQuery_Service.symptomInfo_client import gRPCSymptomInfoQueryServicer
from app.flwr_client import FlowerClinet

import logging
import requests
import configparser

# Create logger
logger = logging.getLogger("main.py")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] main.py - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def retrain_event():
    config = load_config('./app/config.ini')
    apiserver_url = 'http://' + config['ApiServer']['ip'] + ':' + config['ApiServer']['port']
    event_url = f'{apiserver_url}/fl_symptom_retrain_queue/show_current'
    response = requests.get(event_url)
    event = response.json().get('current_retrain')
    return event

def main():
    while True:
        # Connect to the gRPC server to obtain permission to participate in the fl retrain
        # response_servicer = gRPCSymptomInfoQueryServicer()
        # symptom, permission = response_servicer.register_and_getSymptom()
        permission = 'Accepted'
        if permission == 'Accepted':
            logger.info(f'Start federated learning client for symptom {symptom}...')
            flower_client = FlowerClinet(symptom=symptom)
            status = flower_client.start()
            
            # Check federated learning process success or not
            if status == 'Fail':
                continue
        else:
            logger.info('Deneied, wait to more new data input ...')
        
        # Wait the current fl retrain event has finished
        while True:
            # event = retrain_event()
            # if event == None or symptom != event:
            #     break
    
if __name__ == '__main__':
    main()