import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import itertools
import torch.optim as optim
import datetime
from network_energy_saving.model import Actor, DQN
import os
from pymongo import MongoClient
from fastapi import HTTPException, status

prefix = 'ddpg_lstm_rsrp_8f-v22-onnx'
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

class Environment:
    def __init__(self, XY_array, RSRP_array, num_routes=120, time_slots=350):
        self.num_routes = num_routes
        self.time_slots = time_slots
        self.current_route = 0
        self.current_slot = 0
        self.XY_array = XY_array
        self.RSRP_array = RSRP_array
    
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
        print("Env idr:",idr," ,scen:",caseId,end=', \n')

        roomSeed = [0, 1, 2, 3, 4]
        delaySeed = permutations[idr]
        delaySeed2 = permutations[idr2]

        if caseId == 3: # offpeak traffic
            fileName = './mobility/traj_sce2a_arch_40ms_heatmap.npz'
            #print(fileName)
            route = np.load(fileName)
            
            UE_posY=route['UE_posY_at_t']
            UE_posX=route['UE_posX_at_t']
            
            # UE_mobility_id=route['UE_inRoom_ind_at_t']
            # l3RSRP=route['standard_ARMA_L3_RSRP_dB_history_per_UE_per_gNB_at_t']
        
            #print(UE_posX.shape, UE_posY.shape, l3RSRP.shape)
            for ii in range(int(UEgroup*0.5)):
                for tsec in range(routelength):
                    # ueRouteFromFile[ii][tsec][0] = UE_mobility_id[ii][int(tsec*(10*1000/40))]
                    # ueRouteFromFile[ii][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                    # ueRouteFromFile[ii][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                    # ueRouteFromFile[ii][tsec][3] = l3RSRP[ii][0][int(tsec*(10*1000/40))]-rsrpoffset
                    # ueRouteFromFile[ii][tsec][4] = l3RSRP[ii][1][int(tsec*(10*1000/40))]-rsrpoffset
                    # ueRouteFromFile[ii][tsec][5] = l3RSRP[ii][2][int(tsec*(10*1000/40))]-rsrpoffset
                    # ueRouteFromFile[ii][tsec][6] = l3RSRP[ii][3][int(tsec*(10*1000/40))]-rsrpoffset
                    # ueRouteFromFile[ii][tsec][7] = l3RSRP[ii][4][int(tsec*(10*1000/40))]-rsrpoffset

                    l3RSRP = self._get_aodt_rsrp(UE_posX[ii][int(tsec*(10*1000/40))],UE_posY[ii][int(tsec*(10*1000/40))])
                    ueRouteFromFile[ii][tsec][0] = 1 # dummy UE mobility id
                    ueRouteFromFile[ii][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][3] = l3RSRP[0]
                    ueRouteFromFile[ii][tsec][4] = l3RSRP[1]
                    ueRouteFromFile[ii][tsec][5] = l3RSRP[2]
                    ueRouteFromFile[ii][tsec][6] = l3RSRP[3]
                    ueRouteFromFile[ii][tsec][7] = l3RSRP[4]
            
        
        if caseId == 2: # exit
            for routeId in range(5):
                #PATH_sce3_arch_from826toLobby_exit_at_50min
                fileName = './mobility/scen3-new/traj_sce3_arch_from'+roomPrefixs[roomSeed[routeId]]+'toLobby_exit_at_'+delayPrefixs[delaySeed[routeId]]+'min_40ms_heatmap.npz'
                #print(fileName)
                route = np.load(fileName)
                
                UE_posY=route['UE_posY_at_t']
                UE_posX=route['UE_posX_at_t']
                
                UE_mobility_id=route['UE_inRoom_ind_at_t']
                l3RSRP=route['standard_ARMA_L3_RSRP_dB_history_per_UE_per_gNB_at_t']
            
                #print(UE_posX.shape, UE_posY.shape, l3RSRP.shape, UE_mobility_id.shape)
                
                
                for ii in range(UEgroup):
                    for tsec in range(routelength):
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][0] = UE_mobility_id[ii][int(tsec*(10*1000/40))]
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][3] = l3RSRP[ii][0][int(tsec*(10*1000/40))]-rsrpoffset
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][4] = l3RSRP[ii][1][int(tsec*(10*1000/40))]-rsrpoffset
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][5] = l3RSRP[ii][2][int(tsec*(10*1000/40))]-rsrpoffset
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][6] = l3RSRP[ii][3][int(tsec*(10*1000/40))]-rsrpoffset
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][7] = l3RSRP[ii][4][int(tsec*(10*1000/40))]-rsrpoffset

                        l3RSRP = self._get_aodt_rsrp(UE_posX[ii][int(tsec*(10*1000/40))],UE_posY[ii][int(tsec*(10*1000/40))])
                        ueRouteFromFile[ii][tsec][0] = 1 # dummy UE mobility id
                        ueRouteFromFile[ii][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii][tsec][3] = l3RSRP[0]
                        ueRouteFromFile[ii][tsec][4] = l3RSRP[1]
                        ueRouteFromFile[ii][tsec][5] = l3RSRP[2]
                        ueRouteFromFile[ii][tsec][6] = l3RSRP[3]
                        ueRouteFromFile[ii][tsec][7] = l3RSRP[4]
            
            
        if caseId == 1: # peak
            fileName = './mobility/traj_sce2a_arch_40ms_heatmap.npz'
            #print(fileName)
            route = np.load(fileName)
            
            UE_posY=route['UE_posY_at_t']
            UE_posX=route['UE_posX_at_t']
            
            UE_mobility_id=route['UE_inRoom_ind_at_t']
            l3RSRP=route['standard_ARMA_L3_RSRP_dB_history_per_UE_per_gNB_at_t']
        
            #print(UE_posX.shape, UE_posY.shape, l3RSRP.shape)
            for ii in range(UEgroup*5):
                for tsec in range(routelength):
                    # ueRouteFromFile[ii][tsec][0] = UE_mobility_id[ii][int(tsec*(10*1000/40))]
                    # ueRouteFromFile[ii][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                    # ueRouteFromFile[ii][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                    # ueRouteFromFile[ii][tsec][3] = l3RSRP[ii][0][int(tsec*(10*1000/40))]-rsrpoffset
                    # ueRouteFromFile[ii][tsec][4] = l3RSRP[ii][1][int(tsec*(10*1000/40))]-rsrpoffset
                    # ueRouteFromFile[ii][tsec][5] = l3RSRP[ii][2][int(tsec*(10*1000/40))]-rsrpoffset
                    # ueRouteFromFile[ii][tsec][6] = l3RSRP[ii][3][int(tsec*(10*1000/40))]-rsrpoffset
                    # ueRouteFromFile[ii][tsec][7] = l3RSRP[ii][4][int(tsec*(10*1000/40))]-rsrpoffset

                    l3RSRP = self._get_aodt_rsrp(UE_posX[ii][int(tsec*(10*1000/40))],UE_posY[ii][int(tsec*(10*1000/40))])
                    ueRouteFromFile[ii][tsec][0] = 1 # dummy UE mobility id
                    ueRouteFromFile[ii][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                    ueRouteFromFile[ii][tsec][3] = l3RSRP[0]
                    ueRouteFromFile[ii][tsec][4] = l3RSRP[1]
                    ueRouteFromFile[ii][tsec][5] = l3RSRP[2]
                    ueRouteFromFile[ii][tsec][6] = l3RSRP[3]
                    ueRouteFromFile[ii][tsec][7] = l3RSRP[4]
            
        if caseId == 0: # show
            for routeId in range(5):
                #PATH_sce3_arch_from826toLobby_exit_at_50min
                #PATH_sce3_arch_fromLobbyto826_show_at_50min_40ms
                fileName = './mobility/scen1-new/PATH_sce1_arch_fromLobbyto'+roomPrefixs[roomSeed[routeId]]+'_show_at_'+delayPrefixs[delaySeed[routeId]]+'min_40ms.npz'
                #print(fileName)
                route = np.load(fileName)

                UE_posY=route['UE_posY_at_t']
                UE_posX=route['UE_posX_at_t']
                
                UE_mobility_id=route['UE_inRoom_ind_at_t']
                l3RSRP=route['standard_ARMA_L3_RSRP_dB_history_per_UE_per_gNB_at_t']
            
                #print(UE_posX.shape, UE_posY.shape, l3RSRP.shape, UE_mobility_id.shape)
                
                
                for ii in range(UEgroup):
                    for tsec in range(routelength):
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][0] = UE_mobility_id[ii][int(tsec*(10*1000/40))]
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][3] = l3RSRP[ii][0][int(tsec*(10*1000/40))]-rsrpoffset
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][4] = l3RSRP[ii][1][int(tsec*(10*1000/40))]-rsrpoffset
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][5] = l3RSRP[ii][2][int(tsec*(10*1000/40))]-rsrpoffset
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][6] = l3RSRP[ii][3][int(tsec*(10*1000/40))]-rsrpoffset
                        # ueRouteFromFile[ii+routeId*UEgroup][tsec][7] = l3RSRP[ii][4][int(tsec*(10*1000/40))]-rsrpoffset

                        l3RSRP = self._get_aodt_rsrp(UE_posX[ii][int(tsec*(10*1000/40))],UE_posY[ii][int(tsec*(10*1000/40))])
                        ueRouteFromFile[ii][tsec][0] = 1 # dummy UE mobility id
                        ueRouteFromFile[ii][tsec][1] = UE_posX[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii][tsec][2] = UE_posY[ii][int(tsec*(10*1000/40))]
                        ueRouteFromFile[ii][tsec][3] = l3RSRP[0]
                        ueRouteFromFile[ii][tsec][4] = l3RSRP[1]
                        ueRouteFromFile[ii][tsec][5] = l3RSRP[2]
                        ueRouteFromFile[ii][tsec][6] = l3RSRP[3]
                        ueRouteFromFile[ii][tsec][7] = l3RSRP[4]

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

    def _get_aodt_rsrp(self, x, y):

        # calculate the distances from the (x, y) to all AODT samples
        distances = np.sqrt((self.XY_array[:, 0] - x) ** 2 + (self.XY_array[:, 1] - y) ** 2)

        # find the 10 nearest samples
        nearest_indices = np.argsort(distances)[:10]

        # only consider samples within 2 meters
        valid_indices = nearest_indices[distances[nearest_indices] <= 2]
        if len(valid_indices) == 0:
            # if no samples within 2 meters, return a default low RSRP value
            return np.full((TOTAL_BS,), -140.0)

        # Use inverse distance weighting to estimate RSRP values
        weights = 1 / ((distances[nearest_indices] + 1e-6)**2)  # add small value to avoid division by zero
        weights /= np.sum(weights)
        # print(f"Nearest indices: {nearest_indices}, Weights: {weights}")
        # print(f"weighted RSRP: {self.RSRP_array[nearest_indices] * weights[:, np.newaxis]}")
        l3_rsrp = np.sum(self.RSRP_array[nearest_indices] * weights[:, np.newaxis], axis=0)
        # print(f"l3 RSRP: {l3_rsrp}")

        return l3_rsrp
    
class DDPGAgent:
    def __init__(self, actor, critic, tau=0.1):
        self.actor = actor
        self.critic = critic
        self.tau = tau
    
    def select_action(self, state, iter, use_onnx_mode: bool = False):
        """
        - training: use_onnx_mode=False  -> keep gumbel_softmax hard=True (original behavior)
        - inference/export-aligned: use_onnx_mode=True -> deterministic logits -> argmax -> one-hot
        """
        self.actor.update_tau(iter, total_epochs=NUM_EPOCHS)

        num_users = sum(state[0][9])
        if num_users == 0:
            action = torch.zeros(self.action_dim, device="cuda")
            action[0] = 1.0
            return action

        s = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)

        with torch.no_grad():
            if not use_onnx_mode:
                # original behavior (stochastic due to gumbel)
                a = self.actor(s, hard=True).squeeze(0)   # one-hot
                return a
            else:
                # deterministic: logits -> argmax -> one-hot
                logits = self.actor.forward_logits(s).squeeze(0)  # [256]
                idx = int(torch.argmax(logits).item())
                a = torch.zeros(self.action_dim, device="cuda")
                a[idx] = 1.0
                return a
               
class PretrainDataGenerator():
    def __init__(self, db_name: str, collection_name: str, model_version: str):
        self.db_name = db_name
        self.collection_name = collection_name
        self.model_version = model_version
        XY_array, RSRP_array = self.load_aodt_data()
        self.environment = Environment(XY_array, RSRP_array)
        
    def generate_pretrain_data(self, actor, critic, episode):
        agent = DDPGAgent(actor, critic)

        # Start generate training data based on current model and exploration strategy
        state = self.environment.reset()
        done = False
        steps = 0 
        avg_reward = 0.0
        while not done:
            if steps % 100 == 0:
                print(f"Pretrain Data Generation - Episode {episode}, Step {steps}")
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
                self.environment.step(state, env_action, flag)

            # === [DDPG-MOD3] Replay buffer 存「環境實際採用的 env_action」 ===
            data = self.make_doc(step = steps, 
                    state_list = state.tolist(), 
                    action = env_action.detach().cpu().tolist(), 
                    reward = reward, 
                    next_state_list = next_state.tolist(), 
                    done = done, 
                    episode = episode)

            self.connect_mongodb(data)
            avg_reward += reward
            steps += 1
            state = next_state

        return avg_reward / steps

    def utc_now_iso(self):
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def make_doc(self,step: int, state_list, action, reward: float, next_state_list, done: bool, episode: int):

        # ----------------------------
        # Env (MongoDB)
        # ----------------------------
        MONGO_URI = os.environ.get("MONGO_URI", "mongodb://140.113.144.121:27017")
        APP_NAME = os.environ.get("APP_NAME", "NES")       # db name in your example :contentReference[oaicite:1]{index=1}
        PROJECT_ID = os.environ.get("PROJECT_ID", "ED8F")            # collection name in your example :contentReference[oaicite:2]{index=2}
        MODEL_VERSION = os.environ.get("MODEL_VERSION", "0.0.1")

        # ----------------------------
        # Transfer control
        # ----------------------------
        BATCH_STEPS = int(os.environ.get("BATCH_STEPS", "100"))
        POLL_SEC = float(os.environ.get("POLL_SEC", "1.0"))
        RANGE_START = os.environ.get("RANGE_START", "-2h")  # Influx range start for queries
        DATA_INTERVAL_SEC = float(os.environ.get("DATA_INTERVAL_SEC", "1.0"))

        RUN_ID = os.environ.get("RUN_ID", "demo_run_001")
        EPISODE = os.environ.get("EPISODE", "0")  # keep string for tag match

        # Follow the style of your push_data.py :contentReference[oaicite:3]{index=3}
        return {
            "project_id": PROJECT_ID,
            "app_name": APP_NAME,
            "model_version": MODEL_VERSION,
            "input_format": ["state", "next_state"],
            "input": {"state": state_list, "next_state": next_state_list},
            "output_format": ["RU_OnOff"],
            "output": action,
            "KPI_format": ["reward"],
            "KPI": reward,
            "data_interval": DATA_INTERVAL_SEC,
            "timestamp": self.utc_now_iso(),

            # extra metadata (helpful for tracing)
            "run_id": RUN_ID,
            "episode": str(episode),
            "step": int(step),
            "done": bool(done),
        }

    def connect_mongodb(self, data):
        try:
            # Connect to MongoDB
            client = MongoClient("mongodb://mongodb:27017")
            database = client[self.db_name]
            collection = database[self.collection_name]
            # Insert data into MongoDB
            collection.insert_one(data)
            return 'Save data successfully.'
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Post data failed: {str(e)}"
            )
        
    def load_aodt_data(self):
        try:
            # Connect to MongoDB
            client = MongoClient("mongodb://140.113.144.121", 27020)
            database = client[self.db_name]
            collection = database[self.collection_name]
            # Query data from MongoDB
            item_set = collection.find({
                "project_id": self.collection_name,
                "app_name": self.db_name,
                "model_version": self.model_version
            })
            num_items = collection.count_documents({
                "project_id": self.collection_name,
                "app_name": self.db_name,
                "model_version": self.model_version
            })
            # print(f"Retrieved model config: {item_set}")
            # print(f"Data type: {type(item_set)}")
            # print(f"Number of items retrieved: {num_items}")

            XY_array, RSRP_array = self.data_to_numpy(item_set, num_items)

            return XY_array, RSRP_array
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Load AODT data from Common DB failed: {str(e)}"
            )
        
    def data_to_numpy(self, item_set, num_items): 
        # Convert MongoDB data to NumPy array
        idx = 0
        for item in item_set:
            temp_XY = item["input"]["XY"]
            temp_RSRP = item["output"]

            # print(f"temp_state: {np.array(temp_XY)}")
            # print(f"temp_action: {np.array(temp_RSRP)}")

            if idx == 0:
                # Create empty array to store data
                XY_shape = np.array(temp_XY).shape
                RSRP_shape = np.array(temp_RSRP).squeeze().shape

                XY_array = np.zeros((num_items, *XY_shape))
                RSRP_array = np.zeros((num_items, *RSRP_shape))

            XY_array[idx] = np.array(temp_XY)
            RSRP_array[idx] = np.array(temp_RSRP)
            idx += 1

        return XY_array, RSRP_array
