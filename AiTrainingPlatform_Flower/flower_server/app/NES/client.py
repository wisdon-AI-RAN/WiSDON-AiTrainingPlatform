from collections import OrderedDict
from torch.utils.data import Dataset

import time
import logging
import numpy as np
import torch
import torch.nn as nn
import flwr as fl

from .agent import DDPGAgent

logger = logging.getLogger("client.py")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] app/NES/client.py - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Client(fl.client.NumPyClient):
    def __init__(
        self,
        project_id: str,
        version: str,
        app_name: str,
        trainset: Dataset,
        testset: Dataset,
        device: str,
    ):
        self.app_name = app_name
        self.project_id = project_id
        self.version = version
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.fl_type = 'normal'

        # Parameters
        prefix = 'ddpg_lstm_rsrp_8f-v22'
        running_avg_reward = 0.0
        running_avg_rewardb = 0.0
        ACTION_DIM = 256
        NUM_EPOCHS = 1000  # epochs 0..999
        ENABLE_BEST_ACTION_SEARCH = True # toggle brute-force bestAction search


        rsrpoffset = 51-35 # for adjest the rsrp offset
        totalBS = 8
        totalRP = 3629
        totalSample = 1
        shadowing = 1
        caseId = 2
        ueLoc = np.zeros((totalRP,2))
        simData = 350
        step = 0
        numberOfUE = 10*10
        routelength = 360
        UEgroup = 20 
        roomPrefixs = ['801', '814', '816', '824', '826']
        delayPrefixs = ['10', '20', '30', '40', '50']
        ueRouteFromFile = np.zeros((numberOfUE, routelength, 8))
        batchsize = 100
        #print(torch.cuda.current_device())
        torch.set_default_device('cuda')

        # === Model I/O helpers (DDPG) ===
        from pathlib import Path
        # === RU role configuration (centralized) ===
        CAPACITY_RUS = [0, 2, 4]   # controlled by 3 policy bits
        COVERAGE_RUS = [1, 3]      # always ON
        INACTIVE_RUS = [5, 6, 7]   # always OFF; RSRP = -255

        # === Feature layout ===
        N_BASE_FEATS = 8          # 現在 state 的 8 個 feature
        ROLE_FEAT_DIM = 1         # 我們要加一個 role id feature plane
        N_FEATS = N_BASE_FEATS + ROLE_FEAT_DIM  # 8 + 1 = 9

        # === RU role id per index ===
        # 0 = inactive, 1 = capacity, 2 = coverage（之後會縮放到 [0,1]）
        RU_ROLE_ID = np.zeros((totalBS,), dtype=np.float32)
        for ru in CAPACITY_RUS:
            RU_ROLE_ID[ru] = 1.0
        for ru in COVERAGE_RUS:
            RU_ROLE_ID[ru] = 2.0
        # normalize 到 [0,1]，讓網路比較好學
        RU_ROLE_ID = RU_ROLE_ID / 2.0


        RU_ROLES = {"capacity": CAPACITY_RUS, "coverage": COVERAGE_RUS, "inactive": INACTIVE_RUS}
        TOTAL_BS = max([-1] + CAPACITY_RUS + COVERAGE_RUS + INACTIVE_RUS) + 1  # -> 8
        RU_INDEX = list(range(TOTAL_BS))
        # === end RU role configuration ===
        num_episodes=NUM_EPOCHS
        buffer_size=10000
        batch_size=64

        # Initialize model
        self.model = DDPGAgent(state_dim=N_FEATS*10*TOTAL_BS, action_dim=ACTION_DIM)

    def set_parameters(self, parameters):
        # Load model's federated learning parameters with the server's given.            
        actor_net_parameter = parameters["actor"]
        critic_net_parameter = parameters["critic"]
        # actor
        params_dict = zip(self.model.actor.state_dict().keys(), actor_net_parameter)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.actor.load_state_dict(state_dict, strict=True)
        # critic
        params_dict = zip(self.model.critic.state_dict().keys(), critic_net_parameter)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.critic.load_state_dict(state_dict, strict=True)
        # target
        self.model.target_actor.load_state_dict(self.model.actor.state_dict())
        self.model.target_critic.load_state_dict(self.model.critic.state_dict())

    def fit(self, parameters, config):
        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Prepare model training
        loss = self.model.train(self, 
                                train_dataset = self.trainset, 
                                test_dataset = self.testset, 
                                batch_size = batch_size, 
                                epochs = epochs, 
                                gamma=0)

        # Extract model parameters which need to transfer to flower server
        parameters_prime = self.model.get_model_params()
        num_examples_train = len(self.trainset)

        logger.info(f'client starts to transfer parameters : {time.time()}')
        # return parameters_prime, 'flower message', num_examples_train, {"loss" : loss}
        return parameters_prime, num_examples_train, {"loss" : loss}

    def evaluate(self, parameters, config):
        # WIP: Evaluate model on the test dataset here
        return 0.0, 1, {'accuracy': 0.0,
                            'recall': 0.0,
                            'specifity': 0.0,
                            'precision': 0.0}