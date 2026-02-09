"""netdt: A Flower / PyTorch app."""

import sys, os
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from flwr.app import ArrayRecord, MetricRecord, ConfigRecord, RecordDict, Context, Message
from flwr.clientapp import ClientApp
from pymongo import MongoClient

from netdt.model import AERegressor
from netdt.agent import train_fn, test_fn
from netdt.data_loader import DataLoader
from netdt.array_utils import pack_model_arrays, unpack_model_arrays

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

    # Load the model and initialize it with the received weights
    local_model = AERegressor(dropout=0.1)
    local_model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load metadata/dataset from MongoDB/MinIO
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    data_loader = DataLoader(app_name = app_name,
                    project_id = project_id,
                    model_name = model_name,
                    model_version = model_version,
                    mode = mode,
                    dataset_name = dataset_name)
    
    local_file_path = data_loader.query_data(episode=0)

    if mode == "retrain":
        # Load the data
        # Call the retraining function
        train_loss_stage1, train_loss_stage2, datasize = train_fn(
            local_model,
            local_file_path,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            device=device,
        )
    elif mode == "pretrain":
        # Call the pretraining function
        train_loss_stage1, train_loss_stage2, datasize = train_fn(
            local_model,
            local_file_path,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            device=device,
        )

    # Construct and return reply Message
    arrays_record = ArrayRecord(local_model.state_dict())
    metrics = {
        "train_loss_stage1": train_loss_stage1,
        "train_loss_stage2": train_loss_stage2,
        "num-examples": datasize,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": arrays_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
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
    # epochs = msg.content["config"]["epochs"]
    # learning_rate = msg.content["config"]["lr"]
    # batch_size = msg.content["config"]["batch_size"]

    # Load the model and initialize it with the received weights
    local_model = AERegressor(dropout=0.1)
    local_model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load metadata/dataset from MongoDB/MinIO
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    data_loader = DataLoader(app_name = app_name,
                    project_id = project_id,
                    model_name = model_name,
                    model_version = model_version,
                    mode = mode,
                    dataset_name = dataset_name)
    
    local_file_path = data_loader.query_data(episode=0)

    """

    if mode == "retrain":
        # Load the data
        # Call the retraining function
        train_loss_stage1, train_loss_stage2, datasize = train_fn(
            local_model,
            local_file_path,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            device=device,
        )
    elif mode == "pretrain":
        # Call the pretraining function
        train_loss_stage1, train_loss_stage2, datasize = train_fn(
            local_model,
            local_file_path,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            device=device,
        )
    """

    # Construct and return reply Message
    metrics = {
        "train_loss_stage1": [],
        "train_loss_stage2": [],
        "num-examples": 1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)