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

from model import LSTMs1, ResNet8_1D, DQN, Actor

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