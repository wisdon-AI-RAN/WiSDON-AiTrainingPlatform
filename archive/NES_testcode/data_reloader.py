import numpy as np
import torch
from pymongo import MongoClient
from fastapi import HTTPException, status

class DataLoader:
    # def __init__(self, file_path: str):
    #     self.file_path = file_path
    def __init__(self, 
                 app_name: str,
                 project_id: str,
                 model_version: str):
        self.file_path = "./network_energy_saving/ddpg_lstm_rsrp_8f-v22s_case0_20251202/case0_ep0000.npz" # This file path is temporary for testing
        self.db_name = app_name
        self.collection_name = project_id
        self.model_version = model_version

    def load_data(self):
        data = np.load(self.file_path)
        states      = data["states"]       # shape: [T, ...]
        actions     = data["actions"]      # shape: [T, ACTION_DIM]
        rewards     = data["rewards"]      # shape: [T]
        next_states = data["next_states"]  # shape: [T, ...]
        dones       = data["dones"]        # shape: [T]，bool

        # print(f"states shape: {states.shape}")
        # print(f"actions shape: {actions.shape}")
        # print(f"rewards shape: {rewards.shape}")    
        # print(f"next_states shape: {next_states.shape}")
        # print(f"dones shape: {dones.shape}")

        return states, actions, rewards, next_states, dones
    
    def create_dataset(self):
        #===== Preprocess data =====#
        # print("Start data preprocessing...")
        # states, actions, rewards, next_states, dones = self.load_data() # load data from local npz file
        states, actions, rewards, next_states, dones = self.connect_mongodb() # load data from MongoDB
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
            generator=torch.Generator(device='cuda').manual_seed(42)
        )

        # print("Preprocess data successfully.")
        # Print train_dataset information
        # print(f"Training dataset length: {len(train_dataset)}")
        # print(f"Test dataset length: {len(test_dataset)}")
        # print("========================================================")

        return train_dataset, test_dataset
    
    def connect_mongodb(self):
        try:
            # Connect to MongoDB
            client = MongoClient('mongodb://localhost:27019')
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

            states, actions, rewards, next_states, dones = self.data_to_numpy(item_set, num_items)

            # Connect to MongoDB
            client = MongoClient('mongodb://localhost:27017')
            database = client[self.db_name]
            collection = database[self.collection_name]

            # Insert data into MongoDB
            # for item in item_set:
            #     collection.insert_one(item)
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Get training data failed: {str(e)}"
            )
        
    def data_to_numpy(self, item_set, num_items):
        
        # Convert MongoDB data to NumPy array
        idx = 0
        for item in item_set:
            temp_state = item["input"]["state"]
            temp_next_state = item["input"]["next_state"]
            temp_action = item["output"]
            temp_reward = item["KPI"]
            temp_done = item["done"]

            # print(f"temp_state: {np.array(temp_state)}")
            # print(f"temp_action: {np.array(temp_action)}")
            # print(f"temp_reward: {np.array(temp_reward)}")
            # print(f"temp_next_state: {np.array(temp_next_state)}")
            # print(f"temp_done: {np.array(temp_done)}")

            if idx == 0:
                # Create empty array to store data
                state_shape = np.array(temp_state).shape
                action_shape = np.array(temp_action).squeeze().shape
                next_state_shape = np.array(temp_next_state).shape
                reward_shape = np.array(temp_reward).shape
                done_shape = np.array(temp_done).shape

                states = np.zeros((num_items, *state_shape))
                actions = np.zeros((num_items, *action_shape))
                rewards = np.zeros((num_items,))
                next_states =  np.zeros((num_items, *next_state_shape))
                dones = np.zeros((num_items,))

            states[idx] = np.array(temp_state)
            actions[idx] = np.array(temp_action)
            rewards[idx] = np.array(temp_reward)
            next_states[idx] = np.array(temp_next_state)        
            dones[idx] = np.array(temp_done)

            idx += 1

        print(f"states shape: {np.array(states).shape}")
        print(f"actions shape: {np.array(actions).shape}")
        print(f"rewards shape: {np.array(rewards).shape}")
        print(f"next_states shape: {np.array(next_states).shape}")
        print(f"dones shape: {np.array(dones).shape}")
  
        return states, actions, rewards, next_states, dones
        
if __name__ == "__main__":
    data_loader = DataLoader(
        app_name="network_energy_saving",
        project_id="nes_testdata",
        model_version="0.0.1"
    )
    data_loader.connect_mongodb()
    print("Retrieved data from MongoDB.")
