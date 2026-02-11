"""network-energy-saving: A Flower / PyTorch app."""

import sys, os
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from pymongo import MongoClient

from network_energy_saving.model import Actor, DQN
from network_energy_saving.agent import train_fn, test_fn
from network_energy_saving.data_loader import DataLoader
from network_energy_saving.array_utils import pack_model_arrays, unpack_model_arrays


# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Connect to MongoDB and query the current running task
    mongodb_url = os.environ.get("AITRCOMMONDB_URI")
    if mongodb_url is None:
        print("[ERROR CODE 300: FLOWER_INTERNAL_ERROR] AITRCOMMONDB_URI environment variable not set. Cannot connect to MongoDB.")
        return
    client = MongoClient(mongodb_url)
    database = client["TrainingConfig"]
    collection = database["current_task"]
    item = collection.find_one({
        "status": "running"
    })
    project_id = item["project_id"]
    app_name = item["app_name"]
    model_name = item["model_name"]
    model_version = item["model_version"]
    mode = item["mode"]
    dataset_name = item["dataset_name"]

    # Query training parameters
    client = MongoClient(mongodb_url)
    collection_name = f"{project_id}_{app_name}_{model_name}_{model_version}_{mode}_{dataset_name}"
    database = client["TrainingConfig"]
    collection = database[collection_name]
    item = collection.find_one({
        "project_id": project_id,
        "app_name": app_name,
        "model_name": model_name,
        "model_version": model_version,
        "mode": mode,
        "dataset_name": dataset_name,
    })

    # Get training parameters from server
    epochs = msg.content["config"]["epochs"]
    learning_rate = msg.content["config"]["lr"]
    batch_size = msg.content["config"]["batch_size"]

    # Parameters
    action_dim = 256
    # === RU role configuration (centralized) ===
    capacity_rus = [0, 2, 4]   # controlled by 3 policy bits
    coverage_rus = [1, 3]      # always ON
    inactive_rus = [5, 6, 7]   # always OFF; RSRP = -255
    # === Feature layout ===
    n_base_feats = 8          # 現在 state 的 8 個 feature
    role_feat_dim = 1         # 我們要加一個 role id feature plane
    n_feats = n_base_feats + role_feat_dim  # 8 + 1 = 9
    total_bs = max([-1] + capacity_rus + coverage_rus + inactive_rus) + 1  # -> 8
    # === end RU role configuration ===

    # Load the model and initialize it with the received weights
    action_dim = 256
    state_dim = n_feats*10*total_bs
    actor = Actor(state_dim, action_dim)
    critic = DQN()

    actor_sd, critic_sd = unpack_model_arrays(msg.content["arrays"])
    actor.load_state_dict(actor_sd)
    critic.load_state_dict(critic_sd)
    # actor.load_state_dict(msg.content["actor_arrays"].to_torch_state_dict())
    # critic.load_state_dict(msg.content["critic_arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset from MongoDB
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    data_loader = DataLoader(app_name = app_name,
                    project_id = project_id,
                    model_name = model_name,
                    model_version = model_version,
                    mode = mode,
                    dataset_name = dataset_name)
    
    train_dataset, valid_dataset = data_loader.create_dataset(episode=0,split=0,random_seed=42)
 
    if mode == "retrain":
        # Load the data
        # Call the retraining function
        actor_loss, critic_loss = train_fn(
            actor,
            critic,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            gamma=0,
            tau=0.1,
            device=device,
        )
    elif mode == "pretrain":
        # Call the pretraining function
        actor_loss, critic_loss, reward_history = train_fn(
            actor,
            critic,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            gamma=0,
            tau=0.1,
            device=device,
        )

    # Construct and return reply Message
    arrays_record = pack_model_arrays(actor, critic)

    metrics = {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "reward": reward_history,
        "num-examples": len(train_dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": arrays_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Connect to MongoDB and query the current running task
    mongodb_url = os.environ.get("AITRCOMMONDB_URI")
    if mongodb_url is None:
        print("[ERROR CODE 300: FLOWER_INTERNAL_ERROR] AITRCOMMONDB_URI environment variable not set. Cannot connect to MongoDB.")
        return
    client = MongoClient(mongodb_url)
    database = client["TrainingConfig"]
    collection = database["current_task"]
    item = collection.find_one({
        "status": "running"
    })
    project_id = item["project_id"]
    app_name = item["app_name"]
    model_name = item["model_name"]
    model_version = item["model_version"]
    mode = item["mode"]
    dataset_name = item["dataset_name"]

    # Query training parameters
    client = MongoClient(mongodb_url)
    collection_name = f"{project_id}_{app_name}_{model_name}_{model_version}_{mode}_{dataset_name}"
    database = client["TrainingConfig"]
    collection = database[collection_name]
    item = collection.find_one({
        "project_id": project_id,
        "app_name": app_name,
        "model_name": model_name,
        "model_version": model_version,
        "mode": mode,
        "dataset_name": dataset_name,
    })

    # Get training parameters from server
    # epochs = msg.content["config"]["epochs"]
    # learning_rate = msg.content["config"]["lr"]
    # batch_size = msg.content["config"]["batch_size"]

    # Parameters
    action_dim = 256
    # === RU role configuration (centralized) ===
    capacity_rus = [0, 2, 4]   # controlled by 3 policy bits
    coverage_rus = [1, 3]      # always ON
    inactive_rus = [5, 6, 7]   # always OFF; RSRP = -255
    # === Feature layout ===
    n_base_feats = 8          # 現在 state 的 8 個 feature
    role_feat_dim = 1         # 我們要加一個 role id feature plane
    n_feats = n_base_feats + role_feat_dim  # 8 + 1 = 9
    total_bs = max([-1] + capacity_rus + coverage_rus + inactive_rus) + 1  # -> 8
    # === end RU role configuration ===

    # Load the model and initialize it with the received weights
    action_dim = 256
    state_dim = n_feats*10*total_bs
    actor = Actor(state_dim, action_dim)
    critic = DQN()

    actor_sd, critic_sd = unpack_model_arrays(msg.content["arrays"])
    actor.load_state_dict(actor_sd)
    critic.load_state_dict(critic_sd)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """
    # Load the data
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    data_loader = DataLoader(app_name = app_name,
                    project_id = project_id,
                    model_name = model_name,
                    model_version = model_version,
                    mode = mode,
                    dataset_name = dataset_name)
    
    _, test_dataset = data_loader.create_dataset(episode=0,split=0,random_seed=42)

    if mode == "retrain":
        # Call the evaluation function
        actor_loss, critic_loss, eval_value = test_fn(
            actor,
            critic,
            test_dataset,
            batch_size=32,
            gamma=0,
            device=device,
        )
    elif mode == "pretrain":
        # Call the pretraining function
        # actor_loss_history, critic_loss_history, reward_history = pretrain_fn(
        #     actor,
        #     critic,
        #     app_name,
        #     project_id,
        #     model_version = model_version,
        #     batch_size=32,
        #     epochs=epochs,
        #     lr=learning_rate,
        #     gamma=0,
        #     tau=0.1,
        #     device=device,
        # )

        # create dummy metrics for pretraining evaluation
        actor_loss = 1.0
        critic_loss = 1.0   
        eval_value = 1.0
    """

    # Construct and return reply Message
    metrics = {
        "eval_actor_loss": [],
        "eval_critic_loss": [],
        "eval_value": [],
        "num-examples": 1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)