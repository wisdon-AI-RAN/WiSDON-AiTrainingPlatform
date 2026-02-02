import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

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
    
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()
#         self.cnn = LSTMs1()
#         self.resnet = ResNet8_1D(input_channels=1, output_dim=256)  # ResNet processes reshaped LSTM output

#         self.drop = nn.Dropout(0.2)
#         self.fc1 = nn.Linear(64, 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, action_dim)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()  # Using sigmoid for binary output (0 or 1)
#         self.tau_gum = 1.0

#     def forward(self, state, hard=False):
#         #print(state.shape)
#         lstm_out = self.cnn(state)
#         x = self.relu(self.fc1(lstm_out))
#         x = self.drop(x)
#         x = self.relu(self.fc2(x))
#         x = self.drop(x)
        
#         logits = self.fc3(x)
#         logits = torch.tanh(logits) * 5

#         x = F.gumbel_softmax(logits, tau=self.tau_gum, hard=hard, dim=-1)

#         return x
    
#     def update_tau(self, epoch, total_epochs=NUM_EPOCHS, min_tau=0.1, max_tau=4.0):
#         # Piecewise: 0-10% hi, 10-90% linear to lo, 90-100% lo
#         total = int(total_epochs)
#         hi, lo = 4.0, 0.1
#         start_flat = int(0.1 * total)
#         end_flat = int(0.9 * total)
#         t = int(epoch)
#         if t < start_flat:
#             tau = hi
#         elif t < end_flat:
#             frac = (t - start_flat) / max(1, (end_flat - start_flat))
#             tau = hi + (lo - hi) * frac
#         else:
#             tau = lo
#         self.tau_gum = float(max(lo, min(hi, tau)))

#     def gumbel_softmax_sample(self, logits, epoch, total_epochs=3000, min_tau=0.1, max_tau=1.0):
#         """Applies Gumbel-Softmax trick with temperature annealing."""
        
#         # Convert logits to a PyTorch tensor if it's a NumPy array
#         if isinstance(logits, np.ndarray):
#             logits = torch.tensor(logits, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

#         # Compute temperature with clamping
#         temperature = max(min_tau, max_tau - (max_tau - min_tau) * (epoch / total_epochs))  

#         # Generate Gumbel noise
#         noise = torch.rand_like(logits)  # Now logits is a tensor
#         gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        
#         y = logits + gumbel_noise
#         return (y / temperature).sigmoid().round()  # Approximate binary sampling

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
        
    def forward_logits(self, state):
        """
        Deterministic forward for export/inference:
        return logits only (NO gumbel_softmax).
        """
        lstm_out = self.cnn(state)
        x = self.relu(self.fc1(lstm_out))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        logits = self.fc3(x)
        logits = torch.tanh(logits) * 5.0
        return logits
    
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