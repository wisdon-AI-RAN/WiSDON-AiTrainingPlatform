"""netdt: A Flower / PyTorch app."""

import sys, os
import inspect
import torch
import numpy as np
import torch.nn as nn
from typing import List, Tuple, Iterable, Optional
import pickle
from dataclasses import asdict
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pymongo import MongoClient

from netdt.model import AERegressor
from netdt.array_utils import pack_model_arrays, unpack_model_arrays
from netdt.model_loader import ModelRepositoryClient
from netdt.export_onnx import export_onnx

from netdt.data_loader import DataLoader
from netdt.data_preprocessing import data_preprocessing

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

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
    epochs = item["epochs"]
    learning_rate = item["learning_rate"] 
    batch_size = item["batch_size"]

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = AERegressor(dropout=0.1)

    # Pack both models into a single ArrayRecord compatible with FedAvg
    initial_arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=1
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": learning_rate, "epochs": epochs, "batch_size": batch_size}),
        num_rounds=num_rounds,
    )

    
    #========================================================================================
    # This is trick to get the training data. It will be removed in the future when we have a better way to pass the training data to the server.
    data_loader = DataLoader(app_name = app_name,
                    project_id = project_id,
                    model_name = model_name,
                    model_version = model_version,
                    mode = mode,
                    dataset_name = dataset_name)
    
    local_file_path = data_loader.query_data(episode=0)
    x_sim, y_sim_n, x_sim_n, x_meas, y_meas_n, x_meas_n, x_mu, x_std, y_meas_mu, y_meas_std, feature_names, label_name, ru_to_pci = data_preprocessing(local_file_path)
    #========================================================================================

    # Save final model to local
    print("\nSaving final model to local...")
    # update the model version (e.g., 1.0.0 -> 1.0.1) for the new model to be uploaded to the Model Repository
    new_model_version = ".".join(model_version.split(".")[:-1] + [str(int(model_version.split(".")[-1]) + 1)])
    print(f"New model version: {new_model_version}")
    os.makedirs(f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}", exist_ok=True)

    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/global_model.pt")

    # Save model in onnx format
    global_model.load_state_dict(state_dict, strict = True)
    try:
        onnx_path = f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/global_model.onnx"
        export_onnx(global_model, onnx_path, x_mu, x_std, y_meas_mu, y_meas_std)
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")

    # Upload final model to Model Repository
    print("\nUploading final model to Model Repository...")
    model_repository_url = os.environ.get("MODEL_REPOSITORY_URL")
    if model_repository_url is None:
        print("[ERROR CODE 300: FLOWER_INTERNAL_ERROR] MODEL_REPOSITORY_URL environment variable not set. Skipping model upload.")
        return
    client = ModelRepositoryClient(base_url=model_repository_url)

    upload_result = client.upload_onnx_model(
        file_path=f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/global_model.pt",
        project_id=project_id,
        app_name=app_name,
        model_name=model_name,
        version=new_model_version,
        component_name="global_model",
        description=f"Mode:{mode}, Final global model exported to .pt",     
        framework="pytorch"
    )
    print(f"Upload .pt model result: {upload_result}")

    upload_result = client.upload_onnx_model(
        file_path=f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/global_model.onnx",
        project_id=project_id,
        app_name=app_name,
        model_name=model_name,
        version=new_model_version,
        component_name="global_model",
        description=f"Mode:{mode}, Final global model exported to ONNX",     
        framework="pytorch"
    )
    print(f"Upload ONNX model result: {upload_result}")