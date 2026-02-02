from collections import OrderedDict
from torch.utils.data import Dataset

import time
import socket
import logging
import configparser
import torch
import torch.nn as nn
import flwr as fl
from NES.client import Client
from data_loader import DataLoader

logger = logging.getLogger("flwr_client.py")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] flwr_client.py - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

class FlowerClinet():
    def __init__(self, 
                 project_id: str,
                 version: str,
                 app_name: str,):
        
        self.project_id = project_id
        self.version = version
        self.app_name = app_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load client config
        # self.config = load_config('/app/config.ini')
        self.config = load_config('./config.ini')
        
    def start(self):
        trainDataset, testDataset = DataLoader().create_dataset()
        client = Client(project_id=self.project_id,
                        version=self.version,
                        app_name=self.app_name,
                        trainset=trainDataset,
                        testset=testDataset,
                        device=self.device)
        
        # Check the flower server is ip and ready
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            while True:
                try: 
                    sock.connect((self.config['gRPCFlower']['ip'], int(self.config['gRPCFlower']['port'])))
                    logger.info('flower server is up and running !')
                    break
                except Exception as e:
                    logger.error('flower server is not ready ...')
                    time.sleep(3)
        finally:
            sock.close()

        # Start flower client
        try:
            server_address = self.config['gRPCFlower']['ip'] + ':' + self.config['gRPCFlower']['port']
            fl.client.start_numpy_client(server_address=server_address, client=client)
            return 'Success'
        except Exception as e:
            logger.error('flower server closed or disconnected ...')
            return 'Fail'
        
if __name__ == '__main__':
    flower_client = FlowerClinet(project_id='test_project',
                                 version='v1',
                                 app_name='test_app')
    flower_client.start()