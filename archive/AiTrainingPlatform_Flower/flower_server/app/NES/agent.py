import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy  # Import the copy module for deep copy
import matplotlib.pyplot as plt
from .model import LSTMs1, ResNet8_1D, DQN, Actor

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
    def __init__(self, state_dim, action_dim, device, tau=0.1):
        self.action_dim = action_dim
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = DQN().to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.tau = tau
        
    def select_action(self, state, iter):
        self.actor.update_tau(iter, total_epochs=NUM_EPOCHS)
        # state[0] -> # of users
        # state[][0] -> first time slot, state[][9] -> last time slot 
        num_users = sum(state[0][9])
        if num_users == 0:
            action = torch.zeros(self.action_dim).to(self.device)
            action[0] = 1.0  # one-hot for action 0
            return action

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda() # unsqueezed for the batch size
        with torch.no_grad():
            action = self.actor(state, hard=True)  # one-hot vector
            action = action.squeeze(0)

        return action

    def train(self, train_dataset, test_dataset, batch_size, epochs, gamma=0):
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # Create lists to store loss history
        actor_training_loss_history = []
        critic_training_loss_history = []
        actor_validation_loss_history = []
        critic_validation_loss_history = []

        for epoch in range(epochs):
            # Training phase
            self.actor.train()
            self.critic.train()
            actor_loss_list = []
            critic_loss_list = []
            
            for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(train_loader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

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

                # warmup_steps = 100
                # if iter > warmup_steps:
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

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())

            actor_training_loss_history.append(np.mean(actor_loss_list))
            critic_training_loss_history.append(np.mean(critic_loss_list))

            # Validation phase
            self.actor.eval()
            self.critic.eval()
            actor_val_loss_list = []
            critic_val_loss_list = []

            for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(test_loader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

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

                actor_val_loss_list.append(actor_loss.item())
                critic_val_loss_list.append(critic_loss.item())

            actor_validation_loss_history.append(np.mean(actor_val_loss_list))
            critic_validation_loss_history.append(np.mean(critic_val_loss_list))

            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train - Actor: {np.mean(actor_loss_list):.5f}, Critic: {np.mean(critic_loss_list):.5f}")
            print(f"  Valid - Actor: {np.mean(actor_val_loss_list):.5f}, Critic: {np.mean(critic_val_loss_list):.5f}")
            print("========================================================")
            
            # Save model checkpoint
            if epoch % 10 == 0 or epoch == epochs - 1:
                torch.save(self.actor.state_dict(), f'./model/test_actor_epoch_{epoch}.pt')
                torch.save(self.critic.state_dict(), f'./model/test_critic_epoch_{epoch}.pt')
                print(f"Model saved at epoch {epoch}")
                
                # Plot losses
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.semilogy(actor_training_loss_history, label='Training')
                plt.semilogy(actor_validation_loss_history, label='Validation')
                plt.title('Policy Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid()
                
                plt.subplot(1, 2, 2)
                plt.semilogy(critic_training_loss_history, label='Training')
                plt.semilogy(critic_validation_loss_history, label='Validation')
                plt.title('Value Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid()
                
                plt.tight_layout()
                # plt.savefig(f'./pretrain_results/training_loss_lr{learning_rate}_round_{retrain_round}.png')
                plt.savefig(f'./results/training_loss.png')
                plt.close()
        
        return critic_loss.item(), actor_loss.item(), entropy.item() # Ensure the function returns two values
    
    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def get_model_params(self):
        # Returns a model's parameters.
        federated_parameters = []
        for _, val in self.actor.state_dict().items():
            federated_parameters.append(val.cpu().numpy())
        for _, val in self.critic.state_dict().items():
            federated_parameters.append(val.cpu().numpy())
        return federated_parameters