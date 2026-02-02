"""network-energy-saving: A Flower / PyTorch app."""
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from network_energy_saving.model import Actor, DQN
from network_energy_saving.agent import train_fn, test_fn
from network_energy_saving.data_loader import DataLoader
from network_energy_saving.array_utils import (
    pack_model_arrays,
    unpack_model_arrays,
)

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
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
    actor.to(device)
    critic.to(device)
 
    # Load the data
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    data_loader = DataLoader(app_name = context.run_config["app_name"],
                 project_id = context.run_config["project_id"],
                 model_version = context.run_config["model_version"])
    train_dataset, valid_dataset = data_loader.create_dataset()

    # Call the training function
    actor_loss, critic_loss = train_fn(
        actor,
        critic,
        train_dataset,
        valid_dataset,
        batch_size=32,
        epochs=context.run_config["local-epochs"],
        lr=msg.content["config"]["lr"],
        gamma=0,
        tau=0.1,
        device=device,
    )

    # Construct and return reply Message
    arrays_record = pack_model_arrays(actor, critic)

    metrics = {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "num-examples": len(train_dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": arrays_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
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
    actor.to(device)
    critic.to(device)

    # Load the data
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    data_loader = DataLoader(app_name = context.run_config["app_name"],
                 project_id = context.run_config["project_id"],
                 model_version = context.run_config["model_version"])
    _, test_dataset = data_loader.create_dataset()

    # Call the evaluation function
    actor_loss, critic_loss, eval_value = test_fn(
        actor,
        critic,
        test_dataset,
        batch_size=32,
        gamma=0,
        device=device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_actor_loss": actor_loss,
        "eval_critic_loss": critic_loss,
        "eval_value": eval_value,
        "num-examples": len(test_dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)