"""
Save the NES test data to MongoDB for later use in training and evaluation.
"""
from pymongo import MongoClient
import numpy as np
import datetime
from fastapi import HTTPException, status

class DataHandler:
    def __init__(self, 
                 app_name: str,
                 project_id: str,
                 model_version: str):
        self.file_path = "./ddpg_lstm_rsrp_8f-v22s_case0_20251202/case0_ep0000.npz" # This file path is temporary for testing
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

        print("states shape:", states.shape)
        print("actions shape:", actions.shape)
        print("rewards shape:", rewards.shape)
        print("next_states shape:", next_states.shape)
        print("dones shape:", dones.shape)

        return states, actions, rewards, next_states, dones
    
    def make_data_dict(self, state, action, reward, next_state, done):
        data = {
            "project_id": self.collection_name,
            "app_name": self.db_name,
            "model_version": self.model_version,
            "input_format": ["format_1", "format_2", "format_3"],
            "input": {"state":state, "next_state":next_state},
            "output_format": ["RU_OnOff"],
            "output": action,
            "KPI_format": ["throughput"],
            "KPI": reward,
            "data_interval": 1.0,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        return data

    def connect_mongodb(self, data):
        try:
            # Connect to MongoDB
            client = MongoClient("mongodb://140.113.144.121", 27019)
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

if __name__ == "__main__":
    data_handler = DataHandler(
        app_name="network_energy_saving",
        project_id="nes_testdata",
        model_version="0.0.1"
    )

    for ep_idx in range(1):
        data_handler.file_path = f"./ddpg_lstm_rsrp_8f-v22s_case0_20251202/case0_ep{ep_idx:04d}.npz"
        states, actions, rewards, next_states, dones = data_handler.load_data()
        for t in range(1):
            state = states[t].tolist() # np.ndarray -> list
            action = actions[t].tolist() # np.ndarray -> list
            reward = float(rewards[t]) # np.float32 -> float
            next_state = next_states[t].tolist() # np.ndarray -> list
            done = bool(dones[t]) # np.bool_ -> bool

            data_dict = data_handler.make_data_dict(state, action, reward, next_state, done)
            # print(f"Ep {ep_idx} Step {t} Data dict:", data_dict)
            # print(f"state:{data_dict['input']['state'][0][0][0]}")
            data_handler.connect_mongodb(data_dict)