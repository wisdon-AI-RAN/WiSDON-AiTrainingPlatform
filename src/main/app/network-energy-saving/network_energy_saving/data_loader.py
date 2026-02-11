import sys, os
import numpy as np
import torch
from pymongo import MongoClient
from minio import Minio
from fastapi import HTTPException, status

class DataLoader:
    # def __init__(self, file_path: str):
    #     self.file_path = file_path
    def __init__(self, 
                 app_name: str,
                 project_id: str,
                 model_name: str,
                 model_version: str,
                 mode: str,
                 dataset_name: str):
        self.mongodb_URL = os.environ.get("AITRCOMMONDB_URI")
        self.db_name = "TrainingData"
        self.collection_name = "metadata"

        self.project_id = project_id
        self.app_name = app_name
        self.model_name = model_name
        self.model_version = model_version
        # self.model_version = self.model_version.replace(".", "_")
        self.mode = mode
        self.dataset_name = dataset_name
        
        self.minio_URL = os.environ.get("MINIO_URI")
        self.minio_access_key = os.environ.get("MINIO_ACCESS_KEY")
        self.minio_secret_key = os.environ.get("MINIO_SECRET_KEY")

    def load_data(self, file_path):
        data = np.load(file_path)
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
    
    def fake_data(self):
        """
        This is a temporary function to generate fake data for testing the data loading and training pipeline without relying on the actual data in MongoDB. The generated data has the same shape as expected by the model.
        """
        num_data = 1000
        state_shape = (9, 10, 8)  # Example state shape
        action_shape = (256,)      # Example action shape

        states = np.random.rand(num_data, *state_shape).astype(np.float32)
        actions = np.random.rand(num_data, *action_shape).astype(np.float32)
        rewards = np.random.rand(num_data).astype(np.float32)
        next_states = np.random.rand(num_data, *state_shape).astype(np.float32)
        dones = np.random.randint(0, 2, size=(num_data,)).astype(np.float32)

        return states, actions, rewards, next_states, dones
    
    def create_dataset(self, episode, split=0.1, random_seed=42):
        #===== Preprocess data =====#
        # print("Start data preprocessing...")B
        # states, actions, rewards, next_states, dones = self.fake_data() # generate fake data for testing
        self.query_data(episode) # load metadata/dataset from MongoDB/MinIO, and save the dataset to local file system
        states, actions, rewards, next_states, dones = self.load_data(f"./training_data/{self.dataset_name}") # load data from local file system, and convert to numpy array

        # parameters
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
            dones_tensor,
        )

        # Split data using random_split
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [num_training_data, num_test_data],
            generator=torch.Generator().manual_seed(random_seed)
        )

        # print("Preprocess data successfully.")
        # Print train_dataset information
        # print(f"Training dataset length: {len(train_dataset)}")
        # print(f"Test dataset length: {len(test_dataset)}")
        # print("========================================================")

        return train_dataset, test_dataset
    
    def query_data(self, episode):
        try:
            # Connect to MongoDB
            # Use container name instead of localhost when running in Docker
            # client = MongoClient('mongodb://mongodb_compcommondb:27017', serverSelectionTimeoutMS=5000)
            client = MongoClient(self.mongodb_URL, serverSelectionTimeoutMS=5000)
            database = client[self.db_name]
            collection = database[self.collection_name]

            # Query metadata from MongoDB
            item = collection.find_one({
                "project_id": self.project_id,
                "app_name": self.app_name,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "mode": self.mode,
                "dataset_name": self.dataset_name,
            })

            # Query dataset from MinIO
            minio_uri = item["minio_uri"]
            bucket_name = item["minio_bucket"]
            object_name = item["minio_object"]

            client = Minio(
                self.minio_URL, 
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False 
            )

            client.fget_object(
                bucket_name = bucket_name,
                object_name = object_name,
                file_path = f"./training_data/{self.dataset_name}"
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"[ERROR CODE 300: FLOWER_INTERNAL_ERROR] Get training data failed: {str(e)}"
            )
        
    def data_to_numpy(self, item_set, num_items):
        
        # Convert MongoDB data to NumPy array
        idx = 0
        for item in item_set:
            temp_state = item["input"]["state"]
            temp_next_state = item["input"]["next_state"]
            temp_action = item["output"]["action"]
            temp_reward = item["KPI"]["reward"]

            # print(f"temp_state: {np.array(temp_state)}")
            # print(f"temp_action: {np.array(temp_action)}")
            # print(f"temp_reward: {np.array(temp_reward)}")
            # print(f"temp_next_state: {np.array(temp_next_state)}")

            if idx == 0:
                # Create empty array to store data
                state_shape = np.array(temp_state).shape
                action_shape = np.array(temp_action).squeeze().shape
                next_state_shape = np.array(temp_next_state).shape
                reward_shape = np.array(temp_reward).shape

                states = np.zeros((num_items, *state_shape))
                actions = np.zeros((num_items, *action_shape))
                rewards = np.zeros((num_items, *reward_shape))
                next_states =  np.zeros((num_items, *next_state_shape))

            states[idx] = np.array(temp_state)
            actions[idx] = np.array(temp_action)
            rewards[idx] = np.array(temp_reward)
            next_states[idx] = np.array(temp_next_state)        
            idx += 1

        print(f"states shape: {np.array(states).shape}")
        print(f"actions shape: {np.array(actions).shape}")
        print(f"rewards shape: {np.array(rewards).shape}")
        print(f"next_states shape: {np.array(next_states).shape}")
  
        return states, actions, rewards, next_states
        
# if __name__ == "__main__":
#     data_loader = DataLoader(
#         app_name="network_energy_saving",
#         project_id="nes_testdata",
#         model_version="0.0.1"
#     )
#     states, actions, rewards, next_states = data_loader.connect_mongodb()
    # print("Data from MongoDB:", data)