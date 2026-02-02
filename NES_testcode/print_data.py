import numpy as np

ep_file = "./ddpg_lstm_rsrp_8f-v22s_case0_20251202/case0_ep0000.npz"

data = np.load(ep_file)

states      = data["states"]       # shape: [T, ...]
actions     = data["actions"]      # shape: [T, ACTION_DIM]
rewards     = data["rewards"]      # shape: [T]
next_states = data["next_states"]  # shape: [T, ...]
dones       = data["dones"]        # shape: [T]，bool

print("States shape:", states.shape)
print("Actions shape:", actions.shape)
print("Rewards shape:", rewards.shape)
print("Next states shape:", next_states.shape)
print("Dones shape:", dones.shape)