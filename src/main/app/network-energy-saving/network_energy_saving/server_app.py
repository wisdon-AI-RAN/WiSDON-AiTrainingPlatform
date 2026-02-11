"""network-energy-saving: A Flower / PyTorch app."""

import sys, os
import torch
import numpy as np
import torch.nn as nn
from typing import List, Tuple
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pymongo import MongoClient

from network_energy_saving.model import Actor, DQN
from network_energy_saving.array_utils import pack_model_arrays, unpack_model_arrays
from network_energy_saving.model_loader import ModelRepositoryClient
from network_energy_saving.export_onnx import export_onnx

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
    global_actor = Actor(state_dim, action_dim)
    global_critic = DQN()

    # Pack both models into a single ArrayRecord compatible with FedAvg
    initial_arrays = pack_model_arrays(global_actor, global_critic)

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

    # Save final model to disk
    print("\nSaving final model to disk...")
    # update the model version (e.g., 1.0.0 -> 1.0.1) for the new model to be uploaded to the Model Repository
    new_model_version = ".".join(model_version.split(".")[:-1] + [str(int(model_version.split(".")[-1]) + 1)])
    print(f"New model version: {new_model_version}")
    os.makedirs(f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}", exist_ok=True)

    actor_sd, critic_sd = unpack_model_arrays(result.arrays)
    final_global_actor = actor_sd
    final_global_critic = critic_sd
    torch.save(final_global_actor, f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/final_global_actor.pt")
    torch.save(final_global_critic, f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/final_global_critic.pt")

    global_actor.load_state_dict(final_global_actor, strict = True)
    global_critic.load_state_dict(final_global_critic, strict = True)

    # Save model in onnx format (Only actor is needed for inference)
    try:
        onnx_path = f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/final_global_actor_logits.onnx"
        export_onnx(global_actor, onnx_path, n_feats=n_feats, total_bs=total_bs)
        # onnx_path = "./models/final_global_critic_logits.onnx"
        # export_onnx(global_critic, onnx_path, n_feats=n_feats, total_bs=total_bs)
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
        file_path=f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/final_global_actor.pt",
        project_id=project_id,
        app_name=app_name,
        model_name=model_name,
        version=new_model_version,
        component_name="global_actor",
        description=f"Mode:{mode}, Final global_actor exported to .pt",     
        framework="pytorch"
    )
    print(f"Upload .pt model result: {upload_result}")

    upload_result = client.upload_onnx_model(
        file_path=f"./models/{project_id}/{app_name}/{model_name}/{new_model_version}/final_global_actor_logits.onnx",
        project_id=project_id,
        app_name=app_name,
        model_name=model_name,
        version=new_model_version,
        component_name="global_actor_logits",
        description=f"Mode:{mode}, Final global actor logits model exported to ONNX",     
        framework="pytorch"
    )
    print(f"Upload ONNX model result: {upload_result}")

