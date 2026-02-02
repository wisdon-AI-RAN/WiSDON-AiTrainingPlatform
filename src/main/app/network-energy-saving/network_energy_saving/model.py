"""Neural network models for DDPG-based network energy saving."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
ACTION_DIM = 256
NUM_EPOCHS = 3000


class LSTMs1(nn.Module):
    """LSTM-based feature extractor."""

    def __init__(self, input_size=9, hidden_size=64, num_layers=2):
        super(LSTMs1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: [batch, seq_len, features] or [batch, features, seq_len, num_bs]
        if len(x.shape) == 4:
            batch_size, features, seq_len, num_bs = x.shape
            x = x.permute(0, 2, 3, 1)  # [batch, seq_len, num_bs, features]
            x = x.reshape(batch_size, seq_len * num_bs, features)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]  # Return last timestep


class ResNet8_1D(nn.Module):
    """1D ResNet for feature extraction."""

    def __init__(self, input_channels=1, output_dim=256):
        super(ResNet8_1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def _make_layer(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )
        return nn.Sequential(
            ResidualBlock_1D(in_channels, out_channels, stride, downsample),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResidualBlock_1D(nn.Module):
    """1D Residual block."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Actor(nn.Module):
    """Actor network for DDPG-based policy."""

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.cnn = LSTMs1()
        self.resnet = ResNet8_1D(input_channels=1, output_dim=256)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tau_gum = 1.0

    def forward(self, state, hard=False):
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
        """Deterministic forward for export/inference: return logits only (NO gumbel_softmax)."""
        lstm_out = self.cnn(state)
        x = self.relu(self.fc1(lstm_out))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        logits = self.fc3(x)
        logits = torch.tanh(logits) * 5.0
        return logits

    def update_tau(self, epoch, total_epochs=NUM_EPOCHS, min_tau=0.1, max_tau=4.0):
        """Update temperature for Gumbel-Softmax."""
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
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        temperature = max(min_tau, max_tau - (max_tau - min_tau) * (epoch / total_epochs))
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        return (y / temperature).sigmoid().round()


class DQN(nn.Module):
    """DQN Critic network."""

    def __init__(self):
        super(DQN, self).__init__()
        self.cnn = LSTMs1()
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 + ACTION_DIM, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        cnn_out = self.cnn(state)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        x = torch.cat([cnn_out, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        q_value = self.fc3(x)
        return q_value
