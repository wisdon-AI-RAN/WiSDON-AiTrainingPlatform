from collections import OrderedDict

import logging
import configparser
import torch
import torch.nn as nn
import flwr as fl
import numpy as np

from NES.model import Actor, DQN

# Create logger
logger = logging.getLogger("flwr_server.py")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] flwr_server.py - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

class FlowerServer():
    def __init__(self, app_name : str, participants : int):
        self.app_name = app_name
        self.participants = participants

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

        self.state_dim = N_FEATS*10*TOTAL_BS
        self.action_dim = ACTION_DIM
        
        # Load server config
        self.config = load_config('./config.ini')
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.gpu_index = 0
        else:
            self.device = torch.device('cpu')
            self.gpu_index = -1
            
    def fit_config(self, server_round: int):
        """
            Return training configuration dict for each round.

            Keep batch size fixed at 32, perform two rounds of training with one local epoch,
            increase to two local epochs afterwards.
        """
        config = {
            "batch_size": 64,
            "local_epochs": 10,
        }
        return config
        
    def get_model_initial_parameters(self) -> list:
        self.actor_model = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic_model = DQN().to(self.device)
        model_parameters = []
        for _, val in self.actor_model.state_dict().items():
            model_parameters.append(val.cpu().numpy())
        # for _, val in self.critic_model.state_dict().items():
        #     model_parameters.append(val.cpu().numpy())
        return model_parameters
        
    # def store_model_retrained_parameters(self, hist) -> None:
    #     path = f'/app/Model_Repository/{self.symptom}/weight/model.pth.tar'
    #     params_dict = zip(self.model.state_dict().keys(), fl.common.parameters_to_ndarrays(hist.parameters))
    #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #     self.model.load_state_dict(state_dict, strict=True)
    #     torch.save({'state_dict' : self.model.state_dict()}, path)
    
    def start(self):
        model_parameters = self.get_model_initial_parameters()
        
        # Create strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=self.participants,
            min_evaluate_clients=self.participants,
            min_available_clients=self.participants,
            on_fit_config_fn=self.fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        )

        # Start Flower server for num rounds of federated learning
        hist = fl.server.start_server(
            server_address='0.0.0.0:{}'.format(self.config['gRPCFlower']['port']),
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
        )
        
        # If Flower server is ended with some wrong, return directly
        if hist == None:
            logger.error('Flower server closed !')
            return
        
        if self.symptom == 'RL':
            return
        
        # Store final retrained model parameters
        # self.store_model_retrained_parameters(hist=hist)

if __name__ == '__main__':
    flower_server = FlowerServer(app_name='test_app',
                                 participants=1)
    flower_server.start()