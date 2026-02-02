import numpy as np
import torch

class DataLoader:
    # def __init__(self, file_path: str):
    #     self.file_path = file_path
    def __init__(self):
        self.file_path = "./NES/ddpg_lstm_rsrp_8f-v22s_case0_20251202/case0_ep0000.npz"

    def load_data(self):
        data = np.load(self.file_path)
        states      = data["states"]       # shape: [T, ...]
        actions     = data["actions"]      # shape: [T, ACTION_DIM]
        rewards     = data["rewards"]      # shape: [T]
        next_states = data["next_states"]  # shape: [T, ...]
        dones       = data["dones"]        # shape: [T]，bool

        return states, actions, rewards, next_states, dones
    
    def create_dataset(self):
        #===== Preprocess data =====#
        # print("Start data preprocessing...")
        states, actions, rewards, next_states, dones = self.load_data()
        # parameters
        split = 0.1 # 10% for test
        num_data = states.shape[0]
        num_training_data = int(num_data*(1-split))
        num_test_data = num_data - num_training_data

        # Convert to PyTorch tensors
        states_tensor = torch.from_numpy(states).float()
        actions_tensor = torch.from_numpy(actions).float()
        rewards_tensor = torch.from_numpy(rewards).float()
        next_states_tensor = torch.from_numpy(next_states).float()
        dones_tensor = torch.from_numpy(dones).float()

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor
        )

        # Split data using random_split
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [num_training_data, num_test_data],
            generator=torch.Generator().manual_seed(42)
        )

        # print("Preprocess data successfully.")
        # Print train_dataset information
        # print(f"Training dataset length: {len(train_dataset)}")
        # print(f"Test dataset length: {len(test_dataset)}")
        # print("========================================================")

        return train_dataset, test_dataset