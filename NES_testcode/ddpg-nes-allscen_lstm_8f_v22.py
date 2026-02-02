import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import itertools
import torch.optim as optim
import copy  # Import the copy module for deep copy
import time
from datetime import datetime
import scipy.io
import math
import csv

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

# === Role helpers ===
def apply_ru_roles_to_action_from_bits(decoded_bits):
    arr = np.zeros((TOTAL_BS,), dtype=np.float32)
    for ru in CAPACITY_RUS:
        if 0 <= ru < TOTAL_BS:
            arr[ru] = float(decoded_bits[ru])
    for ru in COVERAGE_RUS:
        if 0 <= ru < TOTAL_BS:
            arr[ru] = 1.0
    for ru in INACTIVE_RUS:
        if 0 <= ru < TOTAL_BS:
            arr[ru] = 0.0
    return arr

def validate_roles_array(ruAction):
    ra = np.array(ruAction, dtype=np.float32)
    if len(INACTIVE_RUS):
        assert np.allclose(ra[np.array(INACTIVE_RUS, dtype=int)], 0.0), "inactive RU must be OFF"
    if len(COVERAGE_RUS):
        assert np.allclose(ra[np.array(COVERAGE_RUS, dtype=int)], 1.0), "coverage RU must be ON"
# === End helpers ===



def save_ddpg(agent, file_path, meta: dict | None = None):
    """Save DDPG (actor/critic + targets + optimizers)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "target_actor": agent.target_actor.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "actor_opt": agent.actor_optimizer.state_dict(),
        "critic_opt": agent.critic_optimizer.state_dict(),
        "meta": {
            "prefix": prefix,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            **(meta or {})
        },
    }
    torch.save(payload, str(file_path))
    #print(f"[SAVE] DDPG model saved → {file_path}")

def load_ddpg(agent, file_path, strict: bool = True):
    """Load DDPG checkpoint into existing agent."""
    ckpt = torch.load(file_path, map_location="cuda")
    agent.actor.load_state_dict(ckpt["actor"], strict=strict)
    agent.critic.load_state_dict(ckpt["critic"], strict=strict)
    agent.target_actor.load_state_dict(ckpt["target_actor"], strict=strict)
    agent.target_critic.load_state_dict(ckpt["target_critic"], strict=strict)
    if "actor_opt" in ckpt and "critic_opt" in ckpt:
        agent.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        agent.critic_optimizer.load_state_dict(ckpt["critic_opt"])
    print(f"[LOAD] DDPG model loaded ← {file_path}")



class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class LSTMWithResNet1D(nn.Module):
    def __init__(self, input_size=8*TOTAL_BS, hidden_size=128, num_layers=1, output_size=64):
        super(LSTMWithResNet1D, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.resnet = nn.Sequential(
            BasicBlock1D(1, 16),
            BasicBlock1D(16, 32),
            BasicBlock1D(32, 64),
            nn.AdaptiveAvgPool1d(1),   # Output: [batch, 64, 1]
            nn.Flatten()               # Output: [batch, 64]
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 8*TOTAL_BS, 10)   # [batch, 40, 10]
        x = x.permute(0, 2, 1)             # [batch, 10, 40]

        out, _ = self.lstm(x)              # out: [batch, 10, hidden_size]
        last_time = out[:, -1, :]          # [batch, hidden_size]

        last_time = last_time.unsqueeze(1)  # [batch, 1, hidden_size] = [batch, 1, 128]
        out = self.resnet(last_time)       # [batch, 64]
        return out


class LSTMs1(nn.Module):
    def __init__(self, input_size=N_FEATS*TOTAL_BS, hidden_size=128, num_layers=1, output_size=64):
        super(LSTMs1, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # x: [batch, N_FEATS, 10, TOTAL_BS]
        x = x.view(batch_size, N_FEATS*TOTAL_BS, 10)
        x = x.permute(0, 2, 1)               # [batch, 10, N_FEATS*TOTAL_BS]

        out, (hn, cn) = self.lstm(x)         # out: [1, 10, hidden_size]
        final_output = out[:, -1, :]         # [batch, hidden_size] at last time step

        final_output = self.fc(final_output) # [batch, output_size]
        return final_output
    

# Define 1D ResNet-8 Model
class ResNet8_1D(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(ResNet8_1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(64, 64, num_blocks=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=1, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_dim)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten before fully connected layer
        x = self.fc(x)
        return x

# Environment class
class Environment:
    def __init__(self, num_routes=120, time_slots=350):
        self.num_routes = num_routes
        self.time_slots = time_slots
        self.current_route = 0
        self.current_slot = 0
    
    def reset(self):
        global stepShift, caseId
        stepShift = 0

        inp_list = [0, 1, 2, 3, 4]
        permutations = list(itertools.permutations(inp_list))
        routeData = len(permutations)
        
        idr = random.randint(0, routeData-1)
        caseId = 0
        #print(idr)
        flag = 0
        while flag == 0: 
            #print(idr%5)
            if idr%5 ==0:
                idr = random.randint(0, routeData-1)
                #print(idr)
            else: flag = 1
        
        idr2 = random.randint(0, routeData-1)
        flag = 0
        while flag == 0: 
            #print(idr%5)
            if idr2%5 ==0:
                idr2 = random.randint(0, routeData-1)
                #print(idr)
            else: flag = 1
        #idr = 0
        #
        print("Env idr:",idr," ,scen:",caseId,end=', ')

        roomSeed = [0, 1, 2, 3, 4]
        delaySeed = permutations[idr]
        delaySeed2 = permutations[idr2]

        if caseId == 3: # offpeak traffic
            fileName = 'mobility/PATH_sce2a_arch_40ms.npz'
            #print(fileName)
            route = np.load(fileName)
            
            UE_posY=route['UE_posY_at_t']
            UE_posX=route['UE_posX_at_t']
            
            UE_mobility_id=route['UE_inRoom_ind_at_t']
            l3RSRP=route['standard_ARMA_L3_RSRP_dB_history_per_UE_per_gNB_at_t']
        
            #print(UE_posX.shape, UE_posY.shape, l3RSRP.shape)
            for ii in range(int(UEgroup*0.5)):
                for tsec in range(routelength):
                    ueRouteFromFile[ii][tsec][0] = UE_mobility_id[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][3] = l3RSRP[ii][0][int(tsec*(10*1000/40))]-rsrpoffset
                    ueRouteFromFile[ii][tsec][4] = l3RSRP[ii][1][int(tsec*(10*1000/40))]-rsrpoffset
                    ueRouteFromFile[ii][tsec][5] = l3RSRP[ii][2][int(tsec*(10*1000/40))]-rsrpoffset
                    ueRouteFromFile[ii][tsec][6] = l3RSRP[ii][3][int(tsec*(10*1000/40))]-rsrpoffset
                    ueRouteFromFile[ii][tsec][7] = l3RSRP[ii][4][int(tsec*(10*1000/40))]-rsrpoffset
            
        
        if caseId == 2: # exit
            for routeId in range(5):
                #PATH_sce3_arch_from826toLobby_exit_at_50min
                fileName = 'mobility/scen3-new/PATH_sce3_arch_from'+roomPrefixs[roomSeed[routeId]]+'toLobby_exit_at_'+delayPrefixs[delaySeed[routeId]]+'min_40ms.npz'
                #print(fileName)
                route = np.load(fileName)
                
                UE_posY=route['UE_posY_at_t']
                UE_posX=route['UE_posX_at_t']
                
                UE_mobility_id=route['UE_inRoom_ind_at_t']
                l3RSRP=route['standard_ARMA_L3_RSRP_dB_history_per_UE_per_gNB_at_t']
            
                #print(UE_posX.shape, UE_posY.shape, l3RSRP.shape, UE_mobility_id.shape)
                
                
                for ii in range(UEgroup):
                    for tsec in range(routelength):
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][0] = UE_mobility_id[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][3] = l3RSRP[ii][0][int(tsec*(10*1000/40))]-rsrpoffset
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][4] = l3RSRP[ii][1][int(tsec*(10*1000/40))]-rsrpoffset
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][5] = l3RSRP[ii][2][int(tsec*(10*1000/40))]-rsrpoffset
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][6] = l3RSRP[ii][3][int(tsec*(10*1000/40))]-rsrpoffset
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][7] = l3RSRP[ii][4][int(tsec*(10*1000/40))]-rsrpoffset
            
            
        if caseId == 1: # peak
            fileName = 'mobility/PATH_sce2a_arch_40ms.npz'
            #print(fileName)
            route = np.load(fileName)
            
            UE_posY=route['UE_posY_at_t']
            UE_posX=route['UE_posX_at_t']
            
            UE_mobility_id=route['UE_inRoom_ind_at_t']
            l3RSRP=route['standard_ARMA_L3_RSRP_dB_history_per_UE_per_gNB_at_t']
        
            #print(UE_posX.shape, UE_posY.shape, l3RSRP.shape)
            for ii in range(UEgroup*5):
                for tsec in range(routelength):
                    ueRouteFromFile[ii][tsec][0] = UE_mobility_id[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][3] = l3RSRP[ii][0][int(tsec*(10*1000/40))]-rsrpoffset
                    ueRouteFromFile[ii][tsec][4] = l3RSRP[ii][1][int(tsec*(10*1000/40))]-rsrpoffset
                    ueRouteFromFile[ii][tsec][5] = l3RSRP[ii][2][int(tsec*(10*1000/40))]-rsrpoffset
                    ueRouteFromFile[ii][tsec][6] = l3RSRP[ii][3][int(tsec*(10*1000/40))]-rsrpoffset
                    ueRouteFromFile[ii][tsec][7] = l3RSRP[ii][4][int(tsec*(10*1000/40))]-rsrpoffset
            
        if caseId == 0: # show
            for routeId in range(5):
                #PATH_sce3_arch_from826toLobby_exit_at_50min
                #PATH_sce3_arch_fromLobbyto826_show_at_50min_40ms
                fileName = 'mobility/scen1-new/PATH_sce1_arch_fromLobbyto'+roomPrefixs[roomSeed[routeId]]+'_show_at_'+delayPrefixs[delaySeed[routeId]]+'min_40ms.npz'
                #print(fileName)
                route = np.load(fileName)
                
                UE_posY=route['UE_posY_at_t']
                UE_posX=route['UE_posX_at_t']
                
                UE_mobility_id=route['UE_inRoom_ind_at_t']
                l3RSRP=route['standard_ARMA_L3_RSRP_dB_history_per_UE_per_gNB_at_t']
            
                #print(UE_posX.shape, UE_posY.shape, l3RSRP.shape, UE_mobility_id.shape)
                
                
                for ii in range(UEgroup):
                    for tsec in range(routelength):
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][0] = UE_mobility_id[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][3] = l3RSRP[ii][0][int(tsec*(10*1000/40))]-rsrpoffset
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][4] = l3RSRP[ii][1][int(tsec*(10*1000/40))]-rsrpoffset
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][5] = l3RSRP[ii][2][int(tsec*(10*1000/40))]-rsrpoffset
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][6] = l3RSRP[ii][3][int(tsec*(10*1000/40))]-rsrpoffset
                        ueRouteFromFile[ii+routeId*UEgroup][tsec][7] = l3RSRP[ii][4][int(tsec*(10*1000/40))]-rsrpoffset

        # Initialize state
        ruAction = [1,1,1,1,1,0,0,0]
        length = 10
        ueRoute = np.zeros((numberOfUE, length, 8))
        
        stepShift = 0
        for ii in range(numberOfUE):
            for tsec in range(length):        
                ueRoute[ii][tsec][0] = ueRouteFromFile[ii][stepShift+tsec][0]
                ueRoute[ii][tsec][1] = ueRouteFromFile[ii][stepShift+tsec][1]
                ueRoute[ii][tsec][2] = ueRouteFromFile[ii][stepShift+tsec][2]
                ueRoute[ii][tsec][3] = ueRouteFromFile[ii][stepShift+tsec][3]
                ueRoute[ii][tsec][4] = ueRouteFromFile[ii][stepShift+tsec][4]
                ueRoute[ii][tsec][5] = ueRouteFromFile[ii][stepShift+tsec][5]
                ueRoute[ii][tsec][6] = ueRouteFromFile[ii][stepShift+tsec][6]
                ueRoute[ii][tsec][7] = ueRouteFromFile[ii][stepShift+tsec][7]
        
        state = self._get_init_state(ueRoute, ruAction)
        
        return state
    

    
    def step(self, state, action, flag):
        """
        action:
          - 舊 DDPG：256 維 one-hot（decode_one_hot → 8 bits）
          - 新 MADDPG：8 維 [0/1] 或連續值（每個 RU 一個 bit）
        """
        global stepShift
        done = stepShift >= 350
        if stepShift == 350:
            stepShift = 0

        # --- 解析 action ---

        if isinstance(action, torch.Tensor):
            act_np = action.detach().cpu().numpy()
        else:
            act_np = np.asarray(action, dtype=np.float32)

        # MADDPG：8 維 → 8 個 RU 的 on/off
        if act_np.ndim == 1 and act_np.shape[0] == TOTAL_BS:
            bits = (act_np > 0.5).astype(np.float32)
        else:
            # 相容舊 DDPG：256 維 one-hot → decode 成 8 bits
            bits = np.array(decode_one_hot(act_np), dtype=np.float32)

        # 套用角色（capacity / coverage / inactive）
        ruAction = apply_ru_roles_to_action_from_bits(decoded_bits=bits)
        validate_roles_array(ruAction)



        #print("ruAction=",ruAction)
        
        length = 10
        ueRoute = np.zeros((numberOfUE, length, 8))
        
        stepShift = stepShift+1
        for ii in range(numberOfUE):
            for tsec in range(length):        
                ueRoute[ii][tsec][0] = ueRouteFromFile[ii][stepShift+tsec][0]
                ueRoute[ii][tsec][1] = ueRouteFromFile[ii][stepShift+tsec][1]
                ueRoute[ii][tsec][2] = ueRouteFromFile[ii][stepShift+tsec][2]
                ueRoute[ii][tsec][3] = ueRouteFromFile[ii][stepShift+tsec][3]
                ueRoute[ii][tsec][4] = ueRouteFromFile[ii][stepShift+tsec][4]
                ueRoute[ii][tsec][5] = ueRouteFromFile[ii][stepShift+tsec][5]
                ueRoute[ii][tsec][6] = ueRouteFromFile[ii][stepShift+tsec][6]
                ueRoute[ii][tsec][7] = ueRouteFromFile[ii][stepShift+tsec][7]
        
        

        next_state = self._get_state(state, ueRoute, ruAction)
        #print(state.shape)

        # 計算當前 action 的 reward / capacity / energy
        reward, capacity, totalEnergy = self._calculate_reward(ueRoute, ruAction)

        # 預設情況下，把「最佳」視為目前這個 action
        bestReward = reward
        bestAction = -1
        bestCapacity = capacity
        bestEnergy = totalEnergy

        tflag = 0
        #if stepShift>300: tflag=1; flag=1
        #print(flag)
        if ENABLE_BEST_ACTION_SEARCH and flag==1:
            bestAction, bestReward, bestCapacity, bestEnergy = self._calculate_best_reward(ueRoute, tflag)

        #bestReward = 0.44
        return next_state, reward, capacity, totalEnergy, bestReward, bestAction, bestCapacity, bestEnergy, done

    
       # Count function for each array
    def count_rsrp_ranges(self, arr):
        # Define the bins
        bins = [-140, -100, -90, -80, 0]
        
        if arr.size == 0:
            return np.zeros((4, 1), dtype=int)
        
        hist, _ = np.histogram(arr, bins=bins)
        return hist.reshape((4, 1))
    
    def count_sinr_ranges(self, arr):
        # Define the bins
        bins = [-23, 0, 13, 20, 100]
        
        if arr.size == 0:
            return np.zeros((4, 1), dtype=int)
        
        hist, _ = np.histogram(arr, bins=bins)
        return hist.reshape((4, 1))



    def _get_state(self, state, ueRoute, ruAction):

        ruSwitch = np.zeros((TOTAL_BS,))
        ruPower = np.zeros((TOTAL_BS,))
        for jj in range(totalBS):
            if ruAction[jj] == 0:
                ruSwitch[jj] = 0; ruPower[jj] = 0
            if ruAction[jj] == 1:
                ruSwitch[jj] = 1; ruPower[jj] = 24
        # Implement logic to generate a state (6x10x5 matrix)
        trainingData = np.zeros((N_FEATS, 10, TOTAL_BS))  # 由 8 → N_FEATS
        # # of UE, mean of RSRPs, var of RSRPs, 
        # hist of RSRPs (4)
        # on/off
        # 10 time-slots
        # 5 BS

        # shift 只需要搬舊的 8 個 base feature
        for jj in range(totalBS):
            for tsec in range(10-1):
                for ind in range(N_BASE_FEATS):
                    trainingData[ind][tsec][jj] = state[ind][tsec+1][jj]

        # calculate the last coming sample
        tsec = 9

        validate_roles_array(ruAction)
        MRreport = np.ones((numberOfUE, totalBS+3))*-255
        
        servingTable = np.zeros((TOTAL_BS, 5)) # 5 RUs, {count, rsrp_mean, rsrp_var, sinr_mean, sinr_var}
        MRreportByCell = np.zeros((numberOfUE, 5)) # {serving rsrp}
        MRreportByCell_SINR = np.zeros((numberOfUE, 5)) # {serving sinr}
        countMRreportByCell = np.zeros((TOTAL_BS, )) # {serving, neighbor}

        interference = np.zeros((totalBS,numberOfUE))
        SINR = np.zeros((totalBS,numberOfUE))

        
        for ii in range(numberOfUE):

            # --- MR report fill (generalized to TOTAL_BS) ---
            MRreport[ii, 0:3] = ueRoute[ii, tsec, 0:3]
            # Prepare TOTAL_BS-length RSRP vector, default to -255 for inactive padding
            rsrp_vec = np.full((TOTAL_BS,), -255.0, dtype=float)
            # Number of available RSRP columns in ueRoute (legacy traces may have only 5)
            num_avail = max(0, min(TOTAL_BS, ueRoute.shape[-1] - 3))
            if num_avail > 0:
                rsrp_vec[:num_avail] = np.maximum(ueRoute[ii, tsec, 3:3+num_avail], -140.0)
            # Enforce roles: inactive -> -255
            if len(INACTIVE_RUS):
                rsrp_vec[np.array(INACTIVE_RUS, dtype=int)] = -255.0
            # Write back to MRreport
            MRreport[ii, 3:3+TOTAL_BS] = rsrp_vec
            # --- RSRP array & SINR (vectorized) ---
            rx_dbm = rsrp_vec - 24.0 + ruPower             # [TOTAL_BS]
            off_penalty = 250.0 * (ruSwitch - 1.0)         # [TOTAL_BS]
            rsrpArray = off_penalty + rx_dbm               # [TOTAL_BS]
            noise = 10.0 ** (-100.0 / 10.0)
            lin_pow = ruSwitch * (10.0 ** (rx_dbm / 10.0)) # [TOTAL_BS]
            totalPower = np.sum(lin_pow)
            interf_vec = totalPower - lin_pow
            interference[:, ii] = interf_vec
            SINR[:, ii] = lin_pow / (interf_vec + noise)
            
            sortRsrp = np.sort(rsrpArray)
            sortRsrp = sortRsrp[::-1]
            sortInd = np.argsort(rsrpArray)
            sortInd = sortInd[::-1]
            #print(sortRsrp)
            if MRreport[ii][1]>-100 and MRreport[ii][2] >-100:
                #print(sortInd)
                servingRU=int(sortInd[0])
                #print(servingRU, rsrpArray[servingRU])
                
                MRreportByCell[int(countMRreportByCell[servingRU]),servingRU] = np.max([rsrpArray[servingRU], -140]) # serving rsrp value
                MRreportByCell_SINR[int(countMRreportByCell[servingRU]),servingRU] = np.max([SINR[servingRU][ii], -23]) # serving rsrp value
                countMRreportByCell[servingRU] = countMRreportByCell[servingRU] + 1 # serving rsrp count
                
            
        for jj in range(TOTAL_BS-3):
            # 5 RUs, {count, rsrp_mean, rsrp_var}
            servingTable[jj][0] = countMRreportByCell[jj]
            servingRsrp = MRreportByCell[0:int(countMRreportByCell[jj]), jj]
            servingSinr = MRreportByCell_SINR[0:int(countMRreportByCell[jj]), jj]
            #print(servingRsrp)
            # Process all arrays
            rsrphist = self.count_rsrp_ranges(servingRsrp).flatten()
            #print(rsrphist)
            sinrhist = self.count_sinr_ranges(servingSinr).flatten()
            #print(sinrhist[0],sinrhist[1],sinrhist[2],sinrhist[3])
            #print(MRreportByCell[:,jj])
            if servingTable[jj][0]>0:
                servingTable[jj][1] = np.mean(servingRsrp)
                servingTable[jj][2] = np.var(servingRsrp)**0.5
                servingTable[jj][3] = np.mean(servingSinr)
                servingTable[jj][4] = np.var(servingSinr)**0.5
            if servingTable[jj][0]==0:
                servingTable[jj][1] = -140
                servingTable[jj][2] = 0
                servingTable[jj][3] = -23
                servingTable[jj][4] = 0

            trainingData[0][tsec][jj]=servingTable[jj][0]
            trainingData[1][tsec][jj]=(servingTable[jj][1]+140)/10.0 # normalized to range 0-10
            trainingData[2][tsec][jj]=servingTable[jj][2]
            #trainingData[3][tsec][jj]=(servingTable[jj][3]+23)/10.0 # normalized to range 0-10
            #trainingData[4][tsec][jj]=servingTable[jj][4]
            trainingData[3][tsec][jj]=rsrphist[0]
            trainingData[4][tsec][jj]=rsrphist[1]
            trainingData[5][tsec][jj]=rsrphist[2]
            trainingData[6][tsec][jj]=rsrphist[3]
            #trainingData[9][tsec][jj]=sinrhist[0]
            #trainingData[10][tsec][jj]=sinrhist[1]
            #trainingData[11][tsec][jj]=sinrhist[2]
            #trainingData[12][tsec][jj]=sinrhist[3]
            trainingData[7][tsec][jj]=ruSwitch[jj]

        # ... 上面原本填好 0~7 的 feature 之後，加上 role id feature
        for jj in range(TOTAL_BS):
            trainingData[N_BASE_FEATS, :, jj] = RU_ROLE_ID[jj]
        #print(trainingData)
        state = trainingData #np.random.rand(*self.state_shape)
        
        return state
        

    def _get_init_state(self, ueRoute, ruAction):
        
        ruSwitch = np.zeros((TOTAL_BS,))
        ruPower = np.zeros((TOTAL_BS,))
        for jj in range(totalBS):
            if ruAction[jj] == 0:
                ruSwitch[jj] = 0; ruPower[jj] = 0
            if ruAction[jj] == 1:
                ruSwitch[jj] = 1; ruPower[jj] = 24
        # Implement logic to generate a state (4x10x5 matrix)
        trainingData = np.zeros((N_FEATS, 10, TOTAL_BS)) 
        # # of UE, mean of RSRPs, var of RSRPs, on/off
        for tsec in range(10):
            validate_roles_array(ruAction)
            MRreport = np.ones((numberOfUE, totalBS+3)) * -255.0
        
            servingTable = np.zeros((TOTAL_BS, 5)) # 5 RUs, {count, rsrp_mean, rsrp_var, sinr_mean, sinr_var}
            MRreportByCell = np.zeros((numberOfUE, 5)) # {serving rsrp}
            MRreportByCell_SINR = np.zeros((numberOfUE, 5)) # {serving sinr}
            countMRreportByCell = np.zeros((TOTAL_BS, )) # {serving, neighbor}

            interference = np.zeros((totalBS,numberOfUE))
            SINR = np.zeros((totalBS,numberOfUE))

            
            for ii in range(numberOfUE):

                # --- MR report fill (generalized to TOTAL_BS) ---
                MRreport[ii, 0:3] = ueRoute[ii, tsec, 0:3]
                # Prepare TOTAL_BS-length RSRP vector, default to -255 for inactive padding
                rsrp_vec = np.full((TOTAL_BS,), -255.0, dtype=float)
                # Number of available RSRP columns in ueRoute (legacy traces may have only 5)
                num_avail = max(0, min(TOTAL_BS, ueRoute.shape[-1] - 3))
                if num_avail > 0:
                    rsrp_vec[:num_avail] = np.maximum(ueRoute[ii, tsec, 3:3+num_avail], -140.0)
                # Enforce roles: inactive -> -255
                if len(INACTIVE_RUS):
                    rsrp_vec[np.array(INACTIVE_RUS, dtype=int)] = -255.0
                # Write back to MRreport
                MRreport[ii, 3:3+TOTAL_BS] = rsrp_vec
                # --- RSRP array & SINR (vectorized) ---
                rx_dbm = rsrp_vec - 24.0 + ruPower             # [TOTAL_BS]
                off_penalty = 250.0 * (ruSwitch - 1.0)         # [TOTAL_BS]
                rsrpArray = off_penalty + rx_dbm               # [TOTAL_BS]
                noise = 10.0 ** (-100.0 / 10.0)
                lin_pow = ruSwitch * (10.0 ** (rx_dbm / 10.0)) # [TOTAL_BS]
                totalPower = np.sum(lin_pow)
                interf_vec = totalPower - lin_pow
                interference[:, ii] = interf_vec
                SINR[:, ii] = lin_pow / (interf_vec + noise)
                
                sortRsrp = np.sort(rsrpArray)
                sortRsrp = sortRsrp[::-1]
                sortInd = np.argsort(rsrpArray)
                sortInd = sortInd[::-1]
                #print(sortRsrp)
                if MRreport[ii][1]>-100 and MRreport[ii][2] >-100:
                    #print(sortInd)
                    servingRU=int(sortInd[0])
                    #print(servingRU, rsrpArray[servingRU])
                    
                    MRreportByCell[int(countMRreportByCell[servingRU]),servingRU] = np.max([rsrpArray[servingRU], -140]) # serving rsrp value
                    MRreportByCell_SINR[int(countMRreportByCell[servingRU]),servingRU] = np.max([SINR[servingRU][ii], -23]) # serving rsrp value
                    countMRreportByCell[servingRU] = countMRreportByCell[servingRU] + 1 # serving rsrp count
                
            
        for jj in range(TOTAL_BS-3):
            # 5 RUs, {count, rsrp_mean, rsrp_var}
            servingTable[jj][0] = countMRreportByCell[jj]
            servingRsrp = MRreportByCell[0:int(countMRreportByCell[jj]), jj]
            servingSinr = MRreportByCell_SINR[0:int(countMRreportByCell[jj]), jj]
            #print(servingRsrp)
            # Process all arrays
            rsrphist = self.count_rsrp_ranges(servingRsrp).flatten()
            #print(rsrphist)
            sinrhist = self.count_sinr_ranges(servingSinr).flatten()
            #print(sinrhist[0],sinrhist[1],sinrhist[2],sinrhist[3])
            #print(MRreportByCell[:,jj])
            if servingTable[jj][0]>0:
                servingTable[jj][1] = np.mean(servingRsrp)
                servingTable[jj][2] = np.var(servingRsrp)**0.5
                servingTable[jj][3] = np.mean(servingSinr)
                servingTable[jj][4] = np.var(servingSinr)**0.5
            if servingTable[jj][0]==0:
                servingTable[jj][1] = -140
                servingTable[jj][2] = 0
                servingTable[jj][3] = -23
                servingTable[jj][4] = 0

            trainingData[0][tsec][jj]=servingTable[jj][0]
            trainingData[1][tsec][jj]=(servingTable[jj][1]+140)/10.0 # normalized to range 0-10
            trainingData[2][tsec][jj]=servingTable[jj][2]
            #trainingData[3][tsec][jj]=(servingTable[jj][3]+23)/10.0 # normalized to range 0-10
            #trainingData[4][tsec][jj]=servingTable[jj][4]
            trainingData[3][tsec][jj]=rsrphist[0]
            trainingData[4][tsec][jj]=rsrphist[1]
            trainingData[5][tsec][jj]=rsrphist[2]
            trainingData[6][tsec][jj]=rsrphist[3]
            #trainingData[9][tsec][jj]=sinrhist[0]
            #trainingData[10][tsec][jj]=sinrhist[1]
            #trainingData[11][tsec][jj]=sinrhist[2]
            #trainingData[12][tsec][jj]=sinrhist[3]
            trainingData[7][tsec][jj]=ruSwitch[jj]

        #print(trainingData)
        
        # ... 原本填好 0~7 的 feature
        for jj in range(TOTAL_BS):
            trainingData[N_BASE_FEATS, :, jj] = RU_ROLE_ID[jj]
        state = trainingData #np.random.rand(*self.state_shape)

        return state
    

    def _calculate_reward(self, ueRoute, ruAction):
        ruSwitch = np.zeros((TOTAL_BS,))
        ruPower = np.zeros((TOTAL_BS,))
        for jj in range(totalBS):
            if ruAction[jj] == 0:
                ruSwitch[jj] = 0; ruPower[jj] = 0
            if ruAction[jj] == 1:
                ruSwitch[jj] = 1; ruPower[jj] = 24

        # each loop for 1 sec
        MRreport = np.ones((numberOfUE, totalBS+3)) * -255.0
        
        interference = np.zeros((totalBS,numberOfUE))
        SINR = np.zeros((totalBS,numberOfUE))

        # calculate the last coming sample
        tsec = 9
        AssociTable = np.zeros((TOTAL_BS,)) 
        ueAssoc = np.zeros((numberOfUE,))
        ueSinr = np.zeros((numberOfUE,))
        ueRate = np.zeros((numberOfUE,))

        for ii in range(numberOfUE):

            # --- MR report fill (generalized to TOTAL_BS) ---
            MRreport[ii, 0:3] = ueRoute[ii, tsec, 0:3]
            # Prepare TOTAL_BS-length RSRP vector, default to -255 for inactive padding
            rsrp_vec = np.full((TOTAL_BS,), -255.0, dtype=float)
            # Number of available RSRP columns in ueRoute (legacy traces may have only 5)
            num_avail = max(0, min(TOTAL_BS, ueRoute.shape[-1] - 3))
            if num_avail > 0:
                rsrp_vec[:num_avail] = np.maximum(ueRoute[ii, tsec, 3:3+num_avail], -140.0)
            # Enforce roles: inactive -> -255
            if len(INACTIVE_RUS):
                rsrp_vec[np.array(INACTIVE_RUS, dtype=int)] = -255.0
            # Write back to MRreport
            MRreport[ii, 3:3+TOTAL_BS] = rsrp_vec
            # --- RSRP array & SINR (vectorized) ---
            rx_dbm = rsrp_vec - 24.0 + ruPower             # [TOTAL_BS]
            off_penalty = 250.0 * (ruSwitch - 1.0)         # [TOTAL_BS]
            rsrpArray = off_penalty + rx_dbm               # [TOTAL_BS]
            noise = 10.0 ** (-100.0 / 10.0)
            lin_pow = ruSwitch * (10.0 ** (rx_dbm / 10.0)) # [TOTAL_BS]
            totalPower = np.sum(lin_pow)
            interf_vec = totalPower - lin_pow
            interference[:, ii] = interf_vec
            SINR[:, ii] = lin_pow / (interf_vec + noise)

            sortRsrp = np.sort(rsrpArray)
            sortRsrp = sortRsrp[::-1]
            sortInd = np.argsort(rsrpArray)
            sortInd = sortInd[::-1]

            servingRU=int(sortInd[0])

            ueAssoc[ii] = servingRU
            ueSinr[ii] = SINR[servingRU][ii]
            
            if MRreport[ii][1]>-100 and MRreport[ii][2] >-100:
                AssociTable[servingRU] = AssociTable[servingRU] + 1

        actRuSwitch = ruSwitch
        actRuPower = ruPower
        duCoverage = actRuSwitch[1] + actRuSwitch[3]
        duCapacity = actRuSwitch[0] + actRuSwitch[2] + actRuSwitch[4]
        avgDuCoverage = 0
        avgDuCapacity = 0
            
        ruEnergy = np.zeros((TOTAL_BS,))
        
        
        if duCoverage>0:
            avgDuCoverage = (160-0.2*(90-AssociTable[1]-AssociTable[3]))/duCoverage # PEGA
        if duCapacity>0:
            avgDuCapacity = (160-0.2*(90-AssociTable[0]-AssociTable[2]-AssociTable[4]))/duCapacity # PEGA

        for jj in range(TOTAL_BS):
            ruEnergy[jj] = 0
            if actRuPower[jj] == 24: ruEnergy[jj] = 36.8 - (30-AssociTable[jj])/3
            if actRuPower[jj] == 18: ruEnergy[jj] = 34.0 - (30-AssociTable[jj])/3
        '''     
        if duCoverage>0:
            avgDuCoverage = (263.4)/duCoverage # 2.4 GHz -> 2.2GHz
        if duCapacity>0:
            if duCapacity == 1: avgDuCapacity = 220
            if duCapacity == 2: avgDuCapacity = 263.4/2
            if duCapacity == 3: avgDuCapacity = 287.9/3
            #avgDuCapacity = (287.9)/duCapacity # 2.9 GHz -> 2.2GHz                    
        for jj in range(totalBS):
            if actRuPower[jj] == 24: ruEnergy[jj] = 36.8
            if actRuPower[jj] == 18: ruEnergy[jj] = 34.0
        '''
        ruEnergy[0] = ruEnergy[0] + avgDuCapacity
        ruEnergy[1] = ruEnergy[1] + avgDuCoverage
        ruEnergy[2] = ruEnergy[2] + avgDuCapacity
        ruEnergy[3] = ruEnergy[3] + avgDuCoverage
        ruEnergy[4] = ruEnergy[4] + avgDuCapacity
        
        totalEnergy = 0
        for jj in range(totalBS):
            totalEnergy = totalEnergy + ruSwitch[jj]*ruEnergy[jj]
        

        CapacityState = 0.001
        qos_ratio = 0
        qos_count = 0.001
        gamma_nes = 10.0
        for ii in range(numberOfUE):
            #print(MRreport[ii][1], MRreport[ii][2])
            if MRreport[ii][1]>-100 and MRreport[ii][2] >-100:
                qos_count = qos_count +1
                if MRreport[ii][0] == 0: req = 1 # on hallway
                if MRreport[ii][0] == 1: req = 10 # in-room user
                ueRate[ii] = 100*np.log2(1+ueSinr[ii])/AssociTable[int(ueAssoc[ii])]
                if ueRate[ii]>req: qos_ratio=qos_ratio+1
                req=100
                CapacityState = CapacityState +\
                    np.min([100*np.log2(1+ueSinr[ii])/AssociTable[int(ueAssoc[ii])],req])
                #print(ueRate[ii])
        #CapacityState[action] = CapacityState[action]/totalEnergy
        
        
        qos_ratio = qos_ratio/qos_count
        QoSState = qos_ratio
        EnergyState = totalEnergy
        #CapacityState = qos_ratio - gamma_nes*(totalEnergy/735)
        
        #reward = np.random.randn()  # Example reward
        #reward = CapacityState/1600.0 - gamma_nes*(totalEnergy/735.0) + gamma_nes
        #reward = np.random.randn()  # Example reward
        #if CapacityState>1:
        #    reward = CapacityState/2000.0 - gamma_nes*(totalEnergy/735.0) + gamma_nes
        #if CapacityState<=1:
        #    reward = 1-(totalEnergy/735.0) 
        #reward = gamma_nes*(1-(totalEnergy/504.0))
        
        #if qos_count>=1: 
        #    reward = CapacityState/(qos_count) + gamma_nes*(1-(totalEnergy/504.0))
        #if qos_count==0:
        #    reward = gamma_nes*(1-(totalEnergy/504.0)) 
        # 
        gamma_nes = 0.2
        beta_nes = 1.0
        alpha_nes = 1.0
        eps = 1e-5

        # Clip to prevent log explosion
        capacity = max(CapacityState, eps)
        total_energy = max(totalEnergy, eps)

        num_users = max(qos_count, 1)  # avoid log(0) or log(1)

        reward = np.log(capacity) - gamma_nes * np.log(total_energy) - beta_nes * np.log(num_users)

        # QoS penalty
        capacity_per_user = capacity / num_users
        qos_threshold = 10  # Mbps
        if capacity_per_user < qos_threshold:
            penalty_qos = alpha_nes * np.exp(-capacity_per_user / qos_threshold)
        else:
            penalty_qos = 0

        reward = reward - penalty_qos
        #print("reward = ",reward)
        return reward, CapacityState, totalEnergy
    
    def _calculate_best_reward(self, ueRoute, flag):
        bestReward = -1000
        bestAction = -1
        bestCapacity = None
        bestEnergy = None
        for actionId in range(ACTION_DIM):
            action = np.zeros(ACTION_DIM)
            action[actionId] = 1
            decodedAction = decode_one_hot(torch.tensor(action, dtype=torch.float32))
            ruAction = np.zeros((TOTAL_BS,), dtype=np.float32)
            for ru in CAPACITY_RUS:
                ruAction[ru] = float(decodedAction[ru])
            for ru in COVERAGE_RUS:
                ruAction[ru] = 1.0
            for ru in INACTIVE_RUS:
                ruAction[ru] = 0.0
            ruSwitch = np.zeros((TOTAL_BS,))
            ruPower = np.zeros((TOTAL_BS,))
            for jj in range(totalBS):
                if ruAction[jj] == 0:
                    ruSwitch[jj] = 0; ruPower[jj] = 0
                if ruAction[jj] == 1:
                    ruSwitch[jj] = 1; ruPower[jj] = 24

            # each loop for 1 sec
            MRreport = np.ones((numberOfUE, totalBS+3)) * -255.0
            
            interference = np.zeros((totalBS,numberOfUE))
            SINR = np.zeros((totalBS,numberOfUE))

            # calculate the last coming sample
            tsec = 9

            AssociTable = np.zeros((TOTAL_BS,)) 
            ueAssoc = np.zeros((numberOfUE,))
            ueSinr = np.zeros((numberOfUE,))
            ueRate = np.zeros((numberOfUE,))

            for ii in range(numberOfUE):

                # --- MR report fill (generalized to TOTAL_BS) ---
                MRreport[ii, 0:3] = ueRoute[ii, tsec, 0:3]
                # Prepare TOTAL_BS-length RSRP vector, default to -255 for inactive padding
                rsrp_vec = np.full((TOTAL_BS,), -255.0, dtype=float)
                # Number of available RSRP columns in ueRoute (legacy traces may have only 5)
                num_avail = max(0, min(TOTAL_BS, ueRoute.shape[-1] - 3))
                if num_avail > 0:
                    rsrp_vec[:num_avail] = np.maximum(ueRoute[ii, tsec, 3:3+num_avail], -140.0)
                # Enforce roles: inactive -> -255
                if len(INACTIVE_RUS):
                    rsrp_vec[np.array(INACTIVE_RUS, dtype=int)] = -255.0
                # Write back to MRreport
                MRreport[ii, 3:3+TOTAL_BS] = rsrp_vec
                # --- RSRP array & SINR (vectorized) ---
                rx_dbm = rsrp_vec - 24.0 + ruPower             # [TOTAL_BS]
                off_penalty = 250.0 * (ruSwitch - 1.0)         # [TOTAL_BS]
                rsrpArray = off_penalty + rx_dbm               # [TOTAL_BS]
                noise = 10.0 ** (-100.0 / 10.0)
                lin_pow = ruSwitch * (10.0 ** (rx_dbm / 10.0)) # [TOTAL_BS]
                totalPower = np.sum(lin_pow)
                interf_vec = totalPower - lin_pow
                interference[:, ii] = interf_vec
                SINR[:, ii] = lin_pow / (interf_vec + noise)

                sortRsrp = np.sort(rsrpArray)
                sortRsrp = sortRsrp[::-1]
                sortInd = np.argsort(rsrpArray)
                sortInd = sortInd[::-1]

                servingRU=int(sortInd[0])

                ueAssoc[ii] = servingRU
                ueSinr[ii] = SINR[servingRU][ii]
                
                if MRreport[ii][1]>-100 and MRreport[ii][2] >-100:
                    AssociTable[servingRU] = AssociTable[servingRU] + 1

            actRuSwitch = ruSwitch
            actRuPower = ruPower
            duCoverage = actRuSwitch[1] + actRuSwitch[3]
            duCapacity = actRuSwitch[0] + actRuSwitch[2] + actRuSwitch[4]
            avgDuCoverage = 0
            avgDuCapacity = 0
                
            ruEnergy = np.zeros((TOTAL_BS,))
            
            if duCoverage>0:
                avgDuCoverage = (160-0.2*(90-AssociTable[1]-AssociTable[3]))/duCoverage # PEGA
            if duCapacity>0:
                avgDuCapacity = (160-0.2*(90-AssociTable[0]-AssociTable[2]-AssociTable[4]))/duCapacity # PEGA
            
            for jj in range(TOTAL_BS):
                ruEnergy[jj] = 0
                if actRuPower[jj] == 24: ruEnergy[jj] = 36.8 - (30-AssociTable[jj])/3
                if actRuPower[jj] == 18: ruEnergy[jj] = 34.0 - (30-AssociTable[jj])/3
            '''     
            if duCoverage>0:
                avgDuCoverage = (263.4)/duCoverage # 2.4 GHz -> 2.2GHz
            if duCapacity>0:
                if duCapacity == 1: avgDuCapacity = 220
                if duCapacity == 2: avgDuCapacity = 263.4/2
                if duCapacity == 3: avgDuCapacity = 287.9/3
                #avgDuCapacity = (287.9)/duCapacity # 2.9 GHz -> 2.2GHz                    
            for jj in range(totalBS):
                if actRuPower[jj] == 24: ruEnergy[jj] = 36.8
                if actRuPower[jj] == 18: ruEnergy[jj] = 34.0
            '''
            ruEnergy[0] = ruEnergy[0] + avgDuCapacity
            ruEnergy[1] = ruEnergy[1] + avgDuCoverage
            ruEnergy[2] = ruEnergy[2] + avgDuCapacity
            ruEnergy[3] = ruEnergy[3] + avgDuCoverage
            ruEnergy[4] = ruEnergy[4] + avgDuCapacity
            
            totalEnergy = 0
            for jj in range(totalBS):
                totalEnergy = totalEnergy + ruSwitch[jj]*ruEnergy[jj]
            

            CapacityState = 0.001
            qos_ratio = 0
            qos_count = 0.001
            gamma_nes = 10.0
            for ii in range(numberOfUE):
                #print(MRreport[ii][1], MRreport[ii][2])
                if MRreport[ii][1]>-100 and MRreport[ii][2] >-100:
                    qos_count = qos_count +1
                    if MRreport[ii][0] == 0: req = 1 # on hallway
                    if MRreport[ii][0] == 1: req = 10 # in-room user
                    ueRate[ii] = 100*np.log2(1+ueSinr[ii])/AssociTable[int(ueAssoc[ii])]
                    if ueRate[ii]>req: qos_ratio=qos_ratio+1
                    req=100
                    CapacityState = CapacityState +\
                        np.min([100*np.log2(1+ueSinr[ii])/AssociTable[int(ueAssoc[ii])],req])
                    #print(ueRate[ii])
            #CapacityState[action] = CapacityState[action]/totalEnergy
            
            
            qos_ratio = qos_ratio/qos_count
            QoSState = qos_ratio
            EnergyState = totalEnergy
            #CapacityState = qos_ratio - gamma_nes*(totalEnergy/735)
            
            #reward = np.random.randn()  # Example reward
            #reward = CapacityState/1600.0 - gamma_nes*(totalEnergy/735.0) + gamma_nes

            #reward = np.random.randn()  # Example reward
            #if CapacityState>1:
            #    reward = CapacityState/2000.0 - gamma_nes*(totalEnergy/735.0) + gamma_nes
            #if CapacityState<=1:
            #    reward = 1-(totalEnergy/735.0) 
            #reward = gamma_nes*(1-(totalEnergy/504.0))
            
            #if qos_count>=1: 
            #    reward = CapacityState/(qos_count) + gamma_nes*(1-(totalEnergy/504.0))
            #if qos_count==0:
            #    reward = gamma_nes*(1-(totalEnergy/504.0))  
            
            gamma_nes = 0.2
            beta_nes = 1.0
            alpha_nes = 1.0
            eps = 1e-5

            # Clip to prevent log explosion
            capacity = max(CapacityState, eps)
            total_energy = max(totalEnergy, eps)

            num_users = max(qos_count, 1)  # avoid log(0) or log(1)

            reward = np.log(capacity) - gamma_nes * np.log(total_energy) - beta_nes * np.log(num_users)

            # QoS penalty
            capacity_per_user = capacity / num_users
            qos_threshold = 10  # Mbps
            if capacity_per_user < qos_threshold:
                penalty_qos = alpha_nes * np.exp(-capacity_per_user / qos_threshold)
            else:
                penalty_qos = 0

            reward = reward - penalty_qos

            if flag==1:
                print(actionId, CapacityState, totalEnergy, reward)
            if reward > bestReward:
                bestReward = reward
                bestAction = actionId
                bestCapacity = CapacityState
                bestEnergy = totalEnergy
        return bestAction, bestReward, bestCapacity, bestEnergy

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.max_size = buffer_size
        self.ptr = 0

    def reset(self, buffer_size=None):
        # Reset the buffer and pointer
        if buffer_size:
            self.max_size = buffer_size  # Optional: reset buffer size if provided
        self.buffer = []
        self.ptr = 0

    def add(self, transition):
        # Ensure state, action, and next_state are tensors when adding to buffer
        state, action, reward, next_state, done = transition
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        if isinstance(next_state, np.ndarray):
            next_state = torch.tensor(next_state, dtype=torch.float32)
        
        transition = (state, action, reward, next_state, done)
        
        # If buffer has space, append the transition, else overwrite at `ptr`
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition  # Overwrite the oldest transition
        
        # Increment pointer and wrap around when reaching max_size
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        # Sample a batch from the buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_recent(self, batch_size, recent_ratio=0.1):
        recent_window = int(len(self.buffer) * recent_ratio)
        recent_window = np.max([batch_size, recent_window])
        
        recent_data = self.buffer[-recent_window:]
        indices = np.random.choice(len(recent_data), batch_size, replace=False)
        return [recent_data[i] for i in indices]

    def size(self):
        # Return the number of elements currently in the buffer
        return len(self.buffer)


# === MADDPG: 8RU multi-agent（shared LSTM encoder） ===

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCell(nn.Module):
    """
    單一 RU 的 policy head：輸入 shared embedding，輸出一個 bit（0/1）
    用 Straight-Through Gumbel-Sigmoid 讓離散 action 可以反傳梯度
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.logit = nn.Linear(32, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, s_embed, hard=False):
        # s_embed: [B, embed_dim]
        x = F.relu(self.fc1(s_embed))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        logits = self.logit(x)  # [B, 1]

        if not self.training and hard:
            y = (torch.sigmoid(logits) > 0.5).float()
            return y, logits

        # Gumbel-Sigmoid + Straight-Through
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        y = torch.sigmoid((logits + g))  # tau = 1.0

        if hard:
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y  # ST estimator

        return y, logits


class SharedCritic(nn.Module):
    """
    Centralized Critic：輸入 shared embedding + 8RU action bits → Q(s, a)
    """
    def __init__(self, embed_dim=64, num_agents=TOTAL_BS):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim + num_agents, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, s_embed, actions):
        # s_embed: [B, 64], actions: [B, num_agents]
        x = torch.cat([s_embed, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.q(x)  # [B, 1]


class MultiAgentDDPG:
    """
    shared encoder (LSTMs1) + 8 個 ActorCell + SharedCritic
    """
    def __init__(self, num_agents=TOTAL_BS, gamma=0.99, tau=0.005,
                 actor_lr=1e-4, critic_lr=1e-3):
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau

        # shared encoder
        self.encoder = LSTMs1().to(DEVICE)

        # 8RU actors
        self.actors = nn.ModuleList(
            [ActorCell(embed_dim=64).to(DEVICE) for _ in range(num_agents)]
        )

        # centralized critic
        self.critic = SharedCritic(embed_dim=64, num_agents=num_agents).to(DEVICE)

        # target networks
        self.t_encoder = copy.deepcopy(self.encoder).to(DEVICE)
        self.t_actors = copy.deepcopy(self.actors).to(DEVICE)
        self.t_critic = copy.deepcopy(self.critic).to(DEVICE)

        # optimizers
        self.enc_opt = optim.Adam(self.encoder.parameters(), lr=critic_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_opts = [optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]

    # ---------- acting ----------
    @torch.no_grad()
    def act(self, state, hard=True):
        """
        state: numpy (N_FEATS, 10, TOTAL_BS) 或 torch tensor
        回傳：
          actions: [TOTAL_BS]，每一個元素 ~ {0,1}（RU on/off）
          logits:  [TOTAL_BS]
        """
        if isinstance(state, np.ndarray):
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        elif torch.is_tensor(state):
            if state.dim() == 3:
                s = state.to(DEVICE).unsqueeze(0)
            else:
                s = state.to(DEVICE)
        else:
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        s_embed = self.encoder(s)  # [1, 64]

        acts = []
        logits = []
        for a in self.actors:
            y, lg = a(s_embed, hard=hard)
            acts.append(y)
            logits.append(lg)

        actions = torch.cat(acts, dim=-1).squeeze(0)   # [num_agents]
        logits = torch.cat(logits, dim=-1).squeeze(0)  # [num_agents]
        return actions, logits

    # ---------- training ----------
    def train_step(self, replay_buffer, batch_size):
        if replay_buffer.size() < batch_size:
            return None, None, None

        batch = replay_buffer.sample_recent(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # stack & move to device
        def to_tensor_list(x_list):
            out = []
            for x in x_list:
                if isinstance(x, torch.Tensor):
                    out.append(x.to(DEVICE))
                else:
                    out.append(torch.tensor(x, dtype=torch.float32, device=DEVICE))
            return out

        states = torch.stack(to_tensor_list(states))         # [B, 8, 10, TOTAL_BS]
        actions = torch.stack(to_tensor_list(actions))       # [B, TOTAL_BS]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)  # [B]
        next_states = torch.stack(to_tensor_list(next_states))
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)      # [B]

        B = states.size(0)

        # ----- target Q -----
        with torch.no_grad():
            s2e = self.t_encoder(next_states)  # [B, 64]
            a2_list = []
            for t_actor in self.t_actors:
                y, _ = t_actor(s2e, hard=False)
                a2_list.append(y)
            a2 = torch.cat(a2_list, dim=-1)          # [B, num_agents]
            q_t = self.t_critic(s2e, a2).squeeze(-1) # [B]
            y = rewards + self.gamma * (1.0 - dones) * q_t  # [B]

        # ----- critic & encoder update -----
        se = self.encoder(states)                        # [B, 64]
        q = self.critic(se, actions).squeeze(-1)         # [B]
        critic_loss = F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        self.enc_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            max_norm=5.0
        )
        self.critic_opt.step()
        self.enc_opt.step()

        # ----- actor update -----
        se_pg = self.encoder(states)
        cur_actions = []
        logits_list = []
        for actor in self.actors:
            y, lg = actor(se_pg, hard=False)
            cur_actions.append(y)
            logits_list.append(lg)
        cur_actions = torch.cat(cur_actions, dim=-1)        # [B, num_agents]
        logits = torch.cat(logits_list, dim=-1)             # [B, num_agents]

        q_pi = self.critic(se_pg, cur_actions).squeeze(-1)
        actor_loss = -q_pi.mean()

        # 觀察用 entropy（沒有加進 loss）
        p = torch.sigmoid(logits)
        entropy = (-(p * torch.log(p + 1e-8) +
                     (1 - p) * torch.log(1 - p + 1e-8))).mean()

        for opt in self.actor_opts:
            opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors.parameters(), max_norm=5.0)
        for opt in self.actor_opts:
            opt.step()

        # ----- soft update targets -----
        self._soft_update(self.encoder, self.t_encoder)
        self._soft_update(self.critic, self.t_critic)
        for online, target in zip(self.actors, self.t_actors):
            self._soft_update(online, target)

        return critic_loss.item(), actor_loss.item(), entropy.item()

    def _soft_update(self, online, target):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


def save_maddpg(agent: MultiAgentDDPG, file_path, meta: dict | None = None):
    """
    簡單的 MADDPG checkpoint（encoder + actors + critic + targets + optimizers）
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": agent.encoder.state_dict(),
        "actors": agent.actors.state_dict(),
        "critic": agent.critic.state_dict(),
        "t_encoder": agent.t_encoder.state_dict(),
        "t_actors": agent.t_actors.state_dict(),
        "t_critic": agent.t_critic.state_dict(),
        "enc_opt": agent.enc_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "actor_opts": [opt.state_dict() for opt in agent.actor_opts],
        "meta": {
            "prefix": prefix,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            **(meta or {})
        },
    }
    torch.save(payload, str(file_path))
    print(f"[SAVE] MADDPG model saved → {file_path}")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.cnn = LSTMs1()
        self.resnet = ResNet8_1D(input_channels=1, output_dim=256)  # ResNet processes reshaped LSTM output

        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Using sigmoid for binary output (0 or 1)
        self.tau_gum = 1.0

    def forward(self, state, hard=False):
        #print(state.shape)
        lstm_out = self.cnn(state)
        x = self.relu(self.fc1(lstm_out))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        logits = self.fc3(x)
        logits = torch.tanh(logits) * 5

        x = F.gumbel_softmax(logits, tau=self.tau_gum, hard=hard, dim=-1)

        return x
    
    def update_tau(self, epoch, total_epochs=NUM_EPOCHS, min_tau=0.1, max_tau=4.0):
        # Piecewise: 0-10% hi, 10-90% linear to lo, 90-100% lo
        total = int(total_epochs)
        hi, lo = 4.0, 0.1
        start_flat = int(0.1 * total)
        end_flat = int(0.9 * total)
        t = int(epoch)
        if t < start_flat:
            tau = hi
        elif t < end_flat:
            frac = (t - start_flat) / max(1, (end_flat - start_flat))
            tau = hi + (lo - hi) * frac
        else:
            tau = lo
        self.tau_gum = float(max(lo, min(hi, tau)))

    def gumbel_softmax_sample(self, logits, epoch, total_epochs=3000, min_tau=0.1, max_tau=1.0):
        """Applies Gumbel-Softmax trick with temperature annealing."""
        
        # Convert logits to a PyTorch tensor if it's a NumPy array
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Compute temperature with clamping
        temperature = max(min_tau, max_tau - (max_tau - min_tau) * (epoch / total_epochs))  

        # Generate Gumbel noise
        noise = torch.rand_like(logits)  # Now logits is a tensor
        gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        
        y = logits + gumbel_noise
        return (y / temperature).sigmoid().round()  # Approximate binary sampling

    

def decode_one_hot(one_hot_vector):

    if isinstance(one_hot_vector, np.ndarray):
        one_hot_vector = torch.tensor(one_hot_vector, dtype=torch.float32)
    if isinstance(one_hot_vector, torch.Tensor):
        vec = one_hot_vector.detach().cpu().numpy()
    else:
        vec = np.array(one_hot_vector, dtype=np.float32)
    idx = int(np.argmax(vec))
    bits = [ (idx >> b) & 1 for b in range(TOTAL_BS) ]
    # enforce roles: coverage=1, inactive=0
    for ru in INACTIVE_RUS:
        if 0 <= ru < TOTAL_BS:
            bits[ru] = 0
    for ru in COVERAGE_RUS:
        if 0 <= ru < TOTAL_BS:
            bits[ru] = 1
    return bits

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.cnn = LSTMs1()
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 + ACTION_DIM, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        cnn_out = self.cnn(state)
        #print(cnn_out.shape)
        #print(action.shape)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        #print(action.shape)
        x = torch.cat([cnn_out, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        q_value = self.fc3(x)
        return q_value

class DDPGAgent:
    def __init__(self, state_dim, action_dim, tau=0.1):
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim).cuda()
        self.critic = DQN().cuda()
        self.target_actor = copy.deepcopy(self.actor).cuda()
        self.target_critic = copy.deepcopy(self.critic).cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.tau = tau
    


    def select_action(self, state, iter):
        self.actor.update_tau(iter, total_epochs=NUM_EPOCHS)
        # state[0] -> # of users
        # state[][0] -> first time slot, state[][9] -> last time slot 
        num_users = sum(state[0][9])
        if num_users == 0:
            action = torch.zeros(self.action_dim).to("cuda")
            action[0] = 1.0  # one-hot for action 0
            return action

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda() # unsqueezed for the batch size
        with torch.no_grad():
            action = self.actor(state, hard=True)  # one-hot vector
            action = action.squeeze(0)

        return action

    def train(self, replay_buffer, batch_size, iter, gamma=0):
        if len(replay_buffer.buffer) < batch_size:
            return None, None  # Ensure it returns two values even if training does not happen
        
        batch = replay_buffer.sample_recent(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).cuda()
        actions = torch.stack(actions).cuda()
        rewards = torch.tensor(rewards, dtype=torch.float32).cuda()
        next_states = torch.stack(next_states).cuda()
        dones = torch.tensor(dones, dtype=torch.float32).cuda()

        probs = self.actor(states)

        q_values = self.critic(states, actions).squeeze()
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions).squeeze()
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        critic_loss = nn.MSELoss()(q_values, target_q_values)
        numerator = torch.sum((q_values - target_q_values) ** 2)
        denominator = torch.sum(target_q_values ** 2) + 1e-8
        nloss = numerator / denominator
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #print(nloss)
        critic_loss = nloss
        
        # Debugging: Print critic loss
        #print(f"Critic Loss: {critic_loss.item():.6f}")
        
        # Update Actor
        predicted_actions = self.actor(states)
        #q_values = self.critic(states, predicted_actions)  # must be differentiable
        #advantage = (q_values - q_values.mean()) / (q_values.std() + 1e-5)
        #actor_loss = -advantage.mean()
        
        beta = 0.01  # fixed entropy weight
        q_values = self.critic(states, predicted_actions)
        entropy = -(predicted_actions * torch.log(predicted_actions + 1e-8)).sum(dim=1).mean()
        actor_loss = -q_values.mean() - beta * entropy

        warmup_steps = 100
        if iter > warmup_steps:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            #for name, param in self.actor.named_parameters():
            #    if param.grad is not None:
            #        print(f"{name}: grad norm = {param.grad.norm().item()}")

        #grad_norm = 0
        #for p in self.actor.parameters():
        #    if p.grad is not None:
        #        grad_norm += p.grad.data.norm(2).item() ** 2
        #print("Actor Gradient Norm:", grad_norm ** 0.5)

        #print("Q values:", q_values[:5].tolist())
        #print("Q mean:", q_values.mean().item())
        #print("Q std dev:", q_values.std().item())

        
        # Debugging: Print actor loss
        #print(f"Actor Loss: {actor_loss.item():.10f}")
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        #print(critic_loss.item(), actor_loss.item(), entropy.item())
        
        return critic_loss.item(), actor_loss.item(), entropy.item() # Ensure the function returns two values
    
    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

def train_maddpg(num_episodes=NUM_EPOCHS, buffer_size=10000, batch_size=64):
    date_str = datetime.now().strftime('%Y%m%d')
    run_tag = f"{prefix}_maddpg_{date_str}"
    best_ckpt = f"{run_tag}_best.pth"
    best_avg_reward = float('-inf')

    environment = Environment()
    agent = MultiAgentDDPG(num_agents=TOTAL_BS)
    replay_buffer = ReplayBuffer(buffer_size)

    # 跟 v20 一樣的 episode / qvalue log 檔名（多加 maddpg 標示）
    episode_filename = f"{prefix}-maddpg-{caseId}episode_data.csv"
    qvalue_filename = f"{prefix}-maddpg-{caseId}qvalue_data.csv"

    episode_data = []
    qvalue_data = []

    for episode in range(num_episodes):
        replay_buffer.reset(buffer_size)
        episode_reward = 0.0
        episode_bestReward = 0.0
        episode_entropy = 0.0
        episode_critic_loss = 0.0
        episode_actor_loss = 0.0
        loss_count = 0
        steps = 0

        episode_energy = 0.0
        episode_capacity = 0.0
        episode_bestEnergy = 0.0
        episode_bestCapacity = 0.0

        state = environment.reset()
        done = False

        while not done:
            # 8RU MADDPG 行為：回傳 [8] 的 action tensor
            action_tensor, _ = agent.act(state, hard=True)

            flag = 1  # 保持與 v20 一致（最佳搜尋開關由 ENABLE_BEST_ACTION_SEARCH 控制）
            next_state, reward, capacity, totalEnergy, \
                bestReward, bestAction, bestCapacity, bestEnergy, done = \
                environment.step(state, action_tensor, flag)

            # 最後一個 episode，記 qvalue log（其實是 action / reward trace）
            if episode == num_episodes - 1:
                a_np = action_tensor.detach().cpu().numpy().reshape(-1)
                bits = (a_np > 0.5).astype(int)
                # bits → 對應到原本 0..255 的 action index（純粹為了相容）
                action_idx = 0
                for i, b in enumerate(bits):
                    if b:
                        action_idx |= (1 << i)

                qvalue_data.append([
                    steps,
                    int(action_idx),
                    float(reward),
                    float(capacity),
                    float(bestCapacity),
                    int(bestAction) if isinstance(bestAction, (int, np.integer)) else -1,
                    float(bestReward),
                    float(totalEnergy),
                    float(bestEnergy),
                ])

            # 加進 replay buffer（存 8 維 action）
            replay_buffer.add((state, action_tensor.detach().cpu(), reward, next_state, done))

            # 更新 MADDPG
            critic_loss, actor_loss, entropy = agent.train_step(replay_buffer, batch_size)
            if critic_loss is not None:
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                episode_entropy += entropy
                loss_count += 1

            episode_reward += reward
            episode_bestReward += bestReward
            episode_energy += totalEnergy
            episode_capacity += capacity
            episode_bestEnergy += bestEnergy
            episode_bestCapacity += bestCapacity

            steps += 1
            state = next_state

        # ===== episode 統計 =====
        avg_reward = episode_reward / steps if steps > 0 else 0.0
        avg_bestReward = episode_bestReward / steps if steps > 0 else 0.0
        avg_entropy = episode_entropy / loss_count if loss_count > 0 else 0.0
        avg_critic_loss = episode_critic_loss / loss_count if loss_count > 0 else 0.0
        avg_actor_loss = episode_actor_loss / loss_count if loss_count > 0 else 0.0

        avg_energy = episode_energy / steps if steps > 0 else 0.0
        avg_capacity = episode_capacity / steps if steps > 0 else 0.0
        avg_bestEnergy = episode_bestEnergy / steps if steps > 0 else 0.0
        avg_bestCapacity = episode_bestCapacity / steps if steps > 0 else 0.0

        episode_data.append([
            episode,
            avg_reward,
            avg_bestReward,
            avg_critic_loss,
            avg_actor_loss,
            avg_entropy,
            avg_energy,
            avg_capacity,
            avg_bestEnergy,
            avg_bestCapacity,
        ])

        print(f"[MADDPG] Env scen: {caseId}, Episode {episode}, "
              f"Avg Reward: {avg_reward:.4f}, Best Reward: {avg_bestReward:.4f}, "
              f"Critic Loss: {avg_critic_loss:.6f}, Actor Loss: {avg_actor_loss:.6f}, "
              f"Entropy: {avg_entropy:.6f}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            save_maddpg(agent, best_ckpt, meta={"episode": episode, "avg_reward": avg_reward})

    # --- episode summary CSV ---
    if episode_data:
        header_ep = [
            "Episode",
            "Avg_Reward",
            "Avg_BestReward",
            "Avg_Critic_Loss",
            "Avg_Actor_Loss",
            "Avg_Entropy",
            "Avg_Energy",
            "Avg_Capacity",
            "Avg_BestEnergy",
            "Avg_BestCapacity",
        ]
        with open(episode_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header_ep)
            writer.writerows(episode_data)
        print(f"[SAVE] MADDPG Episode log → {episode_filename}")

    # --- 最後一個 episode 的 step 級 log ---
    if qvalue_data:
        header_q = [
            "Step",
            "Action",
            "Reward",
            "capacity",
            "bestCapacity",
            "BestAction",
            "BestReward",
            "totalEnergy",
            "bestEnergy",
        ]
        with open(qvalue_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header_q)
            writer.writerows(qvalue_data)
        print(f"[SAVE] MADDPG Last-episode trace → {qvalue_filename}")


def train_ddpg(num_episodes=NUM_EPOCHS, buffer_size=10000, batch_size=64):
    date_str = datetime.now().strftime('%Y%m%d')
    run_tag = f"{prefix}_{date_str}"
    best_ckpt = f"{run_tag}_best.pth"
    last_ckpt = f"{run_tag}_last.pth"
    best_avg_reward = float('-inf')
    environment = Environment()
    agent = DDPGAgent(state_dim=N_FEATS*10*TOTAL_BS, action_dim=ACTION_DIM)
    replay_buffer = ReplayBuffer(buffer_size)
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{prefix}_{date_str}.mat"
    episode_data = []
    qvalue_data = []
    
    # 新增：episode / qvalue log 檔名（跟你 v7 一樣的風格）
    episode_filename = f"{prefix}-{caseId}episode_data.csv"
    qvalue_filename = f"{prefix}-{caseId}qvalue_data.csv"
    
    global running_avg_reward
    global running_avg_rewardb
    running_avg_reward = 0.0
    running_avg_rewardb = 0.0
    
    for episode in range(num_episodes):
        replay_buffer.reset(buffer_size)
        episode_reward = 0.0
        episode_entropy = 0.0 
        episode_critic_loss = 0.0  
        episode_actor_loss = 0.0 
        episode_bestReward = 0.0 
        
        episode_energy = 0.0        # 新增：Energy 累積
        episode_capacity = 0.0      # 新增：Capacity 累積
        episode_bestEnergy = 0.0    # 新增：bestEnergy 累積
        episode_bestCapacity = 0.0  # 新增：bestCapacity 累積
        
        loss_count = 0
        steps = 0  
        
        state = environment.reset()
        done = False
        
        
        while not done:
            # 1) Actor 輸出 raw action（256 維 one-hot logits）
            action = agent.select_action(state, episode)

            if episode >= 0:
                flag = 1

            # === [DDPG-MOD1] 從 raw action 建出「policy rule-out 後」的 env_action ===
            # a) 先 decode 成 8 個 RU bits（decode_one_hot 內已 enforce coverage / inactive）
            bits = decode_one_hot(action)                # 長度 TOTAL_BS (=8)，每個是 0/1

            # b) 再把 bits map 回 0~255 的 index（canonical action）
            env_idx = 0
            for b, v in enumerate(bits):
                if v:
                    env_idx |= (1 << b)

            # c) 組回 256 維 one-hot，這個才是「環境與 replay 都使用的 action」
            env_action = torch.zeros_like(action)
            env_action[env_idx] = 1.0

            # === [DDPG-MOD2] env.step 也改成用 env_action，而不是 raw action ===
            next_state, reward, capacity, totalEnergy, bestReward, bestAction, bestCapacity, bestEnergy, done = \
                environment.step(state, env_action, flag)

            # === （保持原本最後一個 episode 的 qvalue log，用 raw action 來記 index 即可）===
            if episode == num_episodes - 1:
                index = int(np.argmax(action.detach().cpu().numpy()))

                # derive RU on/off for action and bestAction
                try:
                    bits_action = decode_one_hot(action)
                except Exception:
                    try:
                        idx_tmp = int(np.argmax(action.detach().cpu().numpy())) if hasattr(action, "detach") else int(np.argmax(action))
                        vec_tmp = np.zeros((ACTION_DIM,), dtype=np.float32)
                        vec_tmp[idx_tmp] = 1.0
                        bits_action = decode_one_hot(vec_tmp)
                    except Exception:
                        bits_action = []
                try:
                    ba_idx = int(bestAction) if not isinstance(bestAction, (list, tuple)) else int(bestAction[0])
                    ba_vec = np.zeros((ACTION_DIM,), dtype=np.float32)
                    ba_vec[ba_idx] = 1.0
                    bits_best = decode_one_hot(ba_vec)
                except Exception:
                    bits_best = []
                    print(f"step {steps}, action: {index}, Reward: {reward:.4f}, "
                          f"bAction: {bestAction}, bReward: {bestReward:.4f}. "
                          f"RU={bits_action}, bRU={bits_best}")

                qvalue_data.append([
                    steps,
                    int(index),          # Action index (0~255)
                    float(reward),
                    float(capacity),
                    float(bestCapacity),
                    int(bestAction),
                    float(bestReward),
                    float(totalEnergy),
                    float(bestEnergy),
                ])

            # === [DDPG-MOD3] Replay buffer 存「環境實際採用的 env_action」 ===
            replay_buffer.add((state, env_action.detach().cpu(), reward, next_state, done))

            # --- 以下維持原本訓練與統計 ---
            if replay_buffer.size() >= batch_size:
                critic_loss, actor_loss, entropy = agent.train(replay_buffer, batch_size, episode)
                if critic_loss is not None and actor_loss is not None:
                    episode_entropy += entropy
                    episode_critic_loss += critic_loss
                    episode_actor_loss += actor_loss
                    loss_count += 1

            episode_reward      += reward
            episode_bestReward  += bestReward
            episode_energy      += totalEnergy
            episode_capacity    += capacity
            episode_bestEnergy  += bestEnergy
            episode_bestCapacity+= bestCapacity

            steps += 1
            state = next_state

        
        
        
        avg_reward = episode_reward / steps if steps > 0 else 0
        avg_entropy = episode_entropy / steps if steps > 0 else 0
        avg_critic_loss = episode_critic_loss / loss_count if loss_count > 0 else 0
        avg_actor_loss = episode_actor_loss / loss_count if loss_count > 0 else 0
        avg_bestReward = episode_bestReward / steps if steps > 0 else 0
        
        avg_energy = episode_energy / steps if steps > 0 else 0
        avg_capacity = episode_capacity / steps if steps > 0 else 0
        avg_bestEnergy = episode_bestEnergy / steps if steps > 0 else 0
        avg_bestCapacity = episode_bestCapacity / steps if steps > 0 else 0
        
        # 存進 episode_data（欄位跟 v7 一致）
        episode_data.append([
            episode,
            avg_reward,
            avg_bestReward,
            avg_critic_loss,
            avg_actor_loss,
            avg_entropy,
            avg_energy,
            avg_capacity,
            avg_bestEnergy,
            avg_bestCapacity,
        ])
        
        #print(loss_count)
        print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Best Reward: {avg_bestReward:.4f}, Critic Loss: {avg_critic_loss:.6f}, Actor Loss: {avg_actor_loss:.6f}, Entropy: {avg_entropy:.6f}, Tau: {agent.actor.tau_gum:.4f}")
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            save_ddpg(agent, best_ckpt, meta={"episode": episode, "avg_reward": avg_reward})
            
    # --- 寫出 episode_data.csv（每個 epoch 的 summary） ---
    if episode_data:
        header_ep = [
            "Episode",
            "Avg_Reward",
            "Avg_BestReward",
            "Avg_Critic_Loss",
            "Avg_Actor_Loss",
            "Avg_Entropy",
            "Avg_Energy",
            "Avg_Capacity",
            "Avg_BestEnergy",
            "Avg_BestCapacity",
        ]
        with open(episode_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header_ep)
            writer.writerows(episode_data)
        print(f"[SAVE] Episode log → {episode_filename}")

    # --- 寫出最後一個 episode 的 step 級 log（qvalue_data.csv） ---
    if qvalue_data:
        header_q = [
            "Step",
            "Action",
            "Reward",
            "capacity",
            "bestCapacity",
            "BestAction",
            "BestReward",
            "totalEnergy",
            "bestEnergy",
        ]
        with open(qvalue_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header_q)
            writer.writerows(qvalue_data)
        print(f"[SAVE] Last-episode qvalue log → {qvalue_filename}")



# Start training
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # 位置引數，可選；不給就用檔案裡原本的 caseId (例如 2)
    parser.add_argument(
        "caseId",
        type=int,
        nargs="?",
        default=caseId,
        help="Scenario id: 0=show, 1=peak, 2=exit, 3=offpeak",
    )
    args = parser.parse_args()
    caseId = args.caseId
    print(f"[INFO] Using caseId={caseId}")
    train_ddpg()
