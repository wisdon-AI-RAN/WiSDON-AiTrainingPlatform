from .Lvsd.lvsd_model import Lvsd_model
from .Lvsd.lvsd_dataset import LvsdDataset
from .Lvsd.contrastive_model.resnet_simclr import ResNet1D18SimCLR as ResNet1D18SimCLR_lvsd

from .Sddd.sddd_model import Sddd_model
from .Sddd.sddd_dataset import SdddDataset
from .Sddd.models.resnet_simclr import ResNet1D18SimCLR as ResNet1D18SimCLR_sddd

from .Pretrained.pretrained_model import pretrained_model
from .Pretrained.pretrained_dataset import *
from .Pretrained.models.resnet_simclr import ResNet1D18SimCLR as ResNet1D18SimCLR_pretrained

from .RL.ddpg import Client

from collections import OrderedDict
from torch.utils.data import Dataset

import time
import socket
import logging
import configparser
import torch
import torch.nn as nn
import flwr as fl

logger = logging.getLogger("flower_client.py")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] flower_client.py - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

class FlowerClinet():
    def __init__(self, symptom : str = 'lvsd'):
        self.symptom = symptom
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        # Load client config
        self.config = load_config('/app/config.ini')
        
    def start(self):
        
        # Load target symptom's dataset
        if self.symptom == 'lvsd':
            trainDataset = LvsdDataset(mode='train')
            testDataset = LvsdDataset(mode='test')
            # client = SymptomClient(symptom=self.symptom, trainset=trainDataset, testset=testDataset, device=self.device).to_client()
            client = SymptomClient(symptom=self.symptom, trainset=trainDataset, testset=testDataset, device=self.device)
            
        if self.symptom == 'sddd':
            trainDataset = SdddDataset(mode='train')
            testDataset = SdddDataset(mode='test')
            # client = SymptomClient(symptom=self.symptom, trainset=trainDataset, testset=testDataset, device=self.device).to_client()
            client = SymptomClient(symptom=self.symptom, trainset=trainDataset, testset=testDataset, device=self.device)
            
        if self.symptom == 'pretrained':
            trainDataset = PretrainedDataset(transform=AugCompose([RandomCropECG(), RandomScaleECG(), RandomDropECG()]))
            client = SymptomClient(symptom=self.symptom, trainset=trainDataset, testset=None, device=self.device)
            
        if self.symptom == 'RL':
            client = SymptomClient(symptom=self.symptom, trainset=None, testset=None, device=self.device)
        
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