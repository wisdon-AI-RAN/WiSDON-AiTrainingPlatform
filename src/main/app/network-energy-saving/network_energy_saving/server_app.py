"""network-energy-saving: A Flower / PyTorch app."""

import sys, os
import inspect
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

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Connect to MongoDB and load training parameters
    mongodb_url = os.environ.get("MONGODB_URL", "mongodb://mongodb:27017")
    client = MongoClient(mongodb_url)
    database = client["Training_Configuration"]
    collection = database["training_conf"]

    # Query data from MongoDB
    item_set = collection.find({
        "project_id": "running"
    })

    for item in item_set:
        project_id = item["project_id"]
        app_name = item["app_name"]
        model_version = item["model_version"]
        mode = item["mode"]
        epochs = item["epochs"]
        learning_rate = item["learning_rate"] 

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = epochs
    lr: float = learning_rate

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
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    # print("\nSaving final model to disk...")
    # state_dict = result.arrays.to_torch_state_dict()
    # torch.save(state_dict, "final_model.pt")
    
    print("\nSaving final model to disk...")
    actor_sd, critic_sd = unpack_model_arrays(result.arrays)
    final_global_actor = actor_sd
    final_global_critic = critic_sd
    torch.save(final_global_actor, "./models/final_global_actor.pt")
    torch.save(final_global_critic, "./models/final_global_critic.pt")

    global_actor.load_state_dict(final_global_actor, strict = True)
    global_critic.load_state_dict(final_global_critic, strict = True)

    # Save model in onnx format (Only actor is needed for inference)
    try:
        onnx_path = "./models/final_global_actor_logits.onnx"
        export_onnx(global_actor, onnx_path, n_feats=n_feats, total_bs=total_bs)
        # onnx_path = "./models/final_global_critic_logits.onnx"
        # export_onnx(global_critic, onnx_path, n_feats=n_feats, total_bs=total_bs)
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")

    # Upload final model to Model Repository
    print("\nUploading final model to Model Repository...")
    model_repository_url = os.environ.get("MODEL_REPOSITORY_URL", "http://ai-model-repository:8000")
    client = ModelRepositoryClient(base_url=model_repository_url)
    upload_result = client.upload_model(
        file_path="./models/final_global_actor_logits.onnx",
        project_id=project_id,
        app_name=app_name,
        version=model_version,
        model_name="global_actor_logits",
        description="Final global actor logits model exported to ONNX",     
        framework="pytorch"
    )
    print(f"Upload actor result: {upload_result}")

    # result = client.upload_model(
    #     file_path="./models/final_global_critic_logits.onnx",
    #     project_id=project_id,
    #     app_name=app_name,
    #     version=model_version,
    #     model_name="global_critic_logits",
    #     description="Final global critic logits model exported to ONNX",     
    #     framework="pytorch"
    # )
    # print(f"Upload critic result: {result}")

def export_onnx(model: Actor, onnx_path: str, opset: int = 18, n_feats: int = 9, total_bs: int = 8) -> None:
    """
    Export deterministic actor logits model to ONNX.
    Input:  state  [B, N_FEATS, 10, TOTAL_BS]
    Output: logits [B, ACTION_DIM]
    """
    model.eval()

    class ActorLogitsWrapper(nn.Module):
        def __init__(self, actor: Actor):
            super().__init__()
            self.actor = actor
        def forward(self, state):
            return self.actor.forward_logits(state)

    orig_device = next(model.parameters()).device
    cpu_model = model.to("cpu").eval()
    wrapper = ActorLogitsWrapper(cpu_model).eval()

    dummy = torch.zeros(1, n_feats, 10, total_bs, dtype=torch.float32, device="cpu")

    export_kwargs = {
        "input_names": ["state"],
        "output_names": ["logits"],
        "dynamic_axes": {"state": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": opset,
    }
    if "use_dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["use_dynamo"] = False

    try:
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            **export_kwargs,
        )
    finally:
        model.to(orig_device)
    print(f"[EXPORT] ONNX saved → {onnx_path}")