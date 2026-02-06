import sys, os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy  # Import the copy module for deep copy
import matplotlib.pyplot as plt
# from network_energy_saving.environment import PretrainDataGenerator
from network_energy_saving.data_loader import DataLoader
from pymongo import MongoClient
import datetime

def train_fn(actor, critic, train_dataset, valid_dataset, batch_size=32, epochs=10, lr=1e-4, gamma=0, tau=0.1, device='cpu'):
        # Initialize models and optimizers
    actor.to(device)
    critic.to(device)
    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator = torch.Generator(device=device),
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator = torch.Generator(device=device),
    )

    # Create lists to store loss history
    actor_training_loss_history = []
    critic_training_loss_history = []
    reward_history = []

    for epoch in range(epochs):
        # Training phase
        actor.train()
        critic.train()
        target_actor.train()
        target_critic.train()
        actor_loss_list = []
        critic_loss_list = []
        reward_list = []
        
        for batch_idx, (states, actions, rewards, next_states) in enumerate(train_loader):
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)

            probs = actor(states)
            q_values = critic(states, actions).squeeze()
            with torch.no_grad():
                next_actions = target_actor(next_states)
                next_q_values = target_critic(next_states, next_actions).squeeze()
                # target_q_values = rewards + gamma * next_q_values * (1 - dones)
                target_q_values = rewards + gamma * next_q_values
            
            critic_loss = nn.MSELoss()(q_values, target_q_values)
            numerator = torch.sum((q_values - target_q_values) ** 2)
            denominator = torch.sum(target_q_values ** 2) + 1e-8
            nloss = numerator / denominator
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            #print(nloss)
            critic_loss = nloss
            
            # Debugging: Print critic loss
            #print(f"Critic Loss: {critic_loss.item():.6f}")
            
            # Update Actor
            predicted_actions = actor(states)
            #q_values = self.critic(states, predicted_actions)  # must be differentiable
            #advantage = (q_values - q_values.mean()) / (q_values.std() + 1e-5)
            #actor_loss = -advantage.mean()
            
            beta = 0.01  # fixed entropy weight
            q_values = critic(states, predicted_actions)
            entropy = -(predicted_actions * torch.log(predicted_actions + 1e-8)).sum(dim=1).mean()
            actor_loss = -q_values.mean() - beta * entropy

            # warmup_steps = 100
            # if iter > warmup_steps:
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            #for name, param in self.actor.named_parameters():
            #    if param.grad is not None:
            #        print(f"{name}: grad norm = {param.grad.norm().item()}")

            #grad_norm = 0
            #for p in self.actor.parameters():
            #    if p.grad is not None:
            #        grad_norm += p.grad.data.norm(2).item() ** 2
            #print("Actor Gradient Norm:", grad_norm ** 0.5)

            #print("Q values:", q_values[:5].tolist())
            #print("Q mean:", q_values.mean().item())
            #print("Q std dev:", q_values.std().item())

            
            # Debugging: Print actor loss
            #print(f"Actor Loss: {actor_loss.item():.10f}")
            
            # Soft update target networks
            soft_update(target_actor, actor, tau)
            soft_update(target_critic, critic, tau)
            #print(critic_loss.item(), actor_loss.item(), entropy.item())

            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            reward_list.append(0)

        actor_training_loss_history.append(np.mean(actor_loss_list))
        critic_training_loss_history.append(np.mean(critic_loss_list))
        reward_history.append(np.mean(reward_list))

        # Update training results to MongoDB
        update_training_results(
            epoch=epoch,
            actor_loss=actor_training_loss_history,
            critic_loss=critic_training_loss_history,   
            reward = reward_history)

        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train - Actor: {np.mean(actor_loss_list):.5f}, Critic: {np.mean(critic_loss_list):.5f}, Reward: {np.mean(reward_list):.5f}")
        print("========================================================")
        
        # Save model checkpoint at local
        # if epoch % 10 == 0 or epoch == epochs - 1:
        #     # Plot losses
        #     plt.figure(figsize=(12, 4))
            
        #     plt.subplot(1, 2, 1)
        #     plt.plot(actor_training_loss_history, label='Training')
        #     plt.title('Policy Loss')
        #     plt.xlabel('Epochs')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     plt.grid()
            
        #     plt.subplot(1, 2, 2)
        #     plt.semilogy(critic_training_loss_history, label='Training')
        #     plt.title('Value Loss')
        #     plt.xlabel('Epochs')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     plt.grid()
            
        #     plt.tight_layout()
        #     plt.savefig(f'./results/training_loss.png')
        #     plt.close()
    
    return np.mean(actor_training_loss_history), np.mean(critic_training_loss_history), np.mean(reward_history)

def test_fn(actor, critic, test_dataset, batch_size, gamma=0, device='cpu'):
    # Initialize models
    actor.to(device)
    critic.to(device)
    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator = torch.Generator(device=device),
    )

    # Validation phase
    actor.eval()
    critic.eval()
    target_actor.eval()
    target_critic.eval()
    actor_test_loss_list = []
    critic_test_loss_list = []
    value_test_list = []

    with torch.no_grad():
        for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(test_loader):
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            probs = actor(states)

            q_values = critic(states, actions).squeeze()
            next_actions = target_actor(next_states)
            next_q_values = target_critic(next_states, next_actions).squeeze()
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            critic_loss = nn.MSELoss()(q_values, target_q_values)
            numerator = torch.sum((q_values - target_q_values) ** 2)
            denominator = torch.sum(target_q_values ** 2) + 1e-8
            nloss = numerator / denominator
            #print(nloss)
            critic_loss = nloss
            
            # Debugging: Print critic loss
            #print(f"Critic Loss: {critic_loss.item():.6f}")
            
            # Update Actor
            predicted_actions = actor(states)
            #q_values = self.critic(states, predicted_actions)  # must be differentiable
            #advantage = (q_values - q_values.mean()) / (q_values.std() + 1e-5)
            #actor_loss = -advantage.mean()
            
            beta = 0.01  # fixed entropy weight
            q_values = critic(states, predicted_actions)
            entropy = -(predicted_actions * torch.log(predicted_actions + 1e-8)).sum(dim=1).mean()
            actor_loss = -q_values.mean() - beta * entropy
            #grad_norm = 0
            #for p in self.actor.parameters():
            #    if p.grad is not None:
            #        grad_norm += p.grad.data.norm(2).item() ** 2
            #print("Actor Gradient Norm:", grad_norm ** 0.5)

            #print("Q values:", q_values[:5].tolist())
            #print("Q mean:", q_values.mean().item())
            #print("Q std dev:", q_values.std().item())
            
            # Debugging: Print actor loss
            #print(f"Actor Loss: {actor_loss.item():.10f}")

            actor_test_loss_list.append(actor_loss.item())
            critic_test_loss_list.append(critic_loss.item())
            value_test_list.append(q_values.mean().item())

    return np.mean(actor_test_loss_list), np.mean(critic_test_loss_list), np.mean(value_test_list)

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# def pretrain_fn(actor, 
#                 critic, 
#                 db_name: str, 
#                 collection_name: str, 
#                 model_version: str,
#                 batch_size=32, 
#                 epochs=10, 
#                 lr=1e-4, 
#                 gamma=0, 
#                 tau=0.1, 
#                 device='cpu'):
    
#     # Initialize models and optimizers
#     actor.to(device)
#     critic.to(device)
#     target_actor = copy.deepcopy(actor).to(device)
#     target_critic = copy.deepcopy(critic).to(device)
#     actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
#     critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
#     env = PretrainDataGenerator(db_name, collection_name, model_version)
#     data_loader = DataLoader(db_name, collection_name, model_version)

#     # Create lists to store loss history
#     actor_training_loss_history = []
#     critic_training_loss_history = []
#     actor_validation_loss_history = []
#     critic_validation_loss_history = []

#     reward_history = []

#     for epoch in range(epochs):
#         # Create new datasets for each epoch
#         actor.eval()
#         critic.eval()
#         avg_reward = env.generate_pretrain_data(actor, critic, epoch)

#         # load data from database
#         train_dataset, valid_dataset = data_loader.create_dataset(epoch, split=0)

#         # Create data loaders
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=0,
#             generator = torch.Generator(device=device),
#         )

#         valid_loader = torch.utils.data.DataLoader(
#             valid_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=0,
#             generator = torch.Generator(device=device),
#         )

#         # Training phase
#         actor.train()
#         critic.train()
#         target_actor.train()
#         target_critic.train()
#         actor_loss_list = []
#         critic_loss_list = []
        
#         for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(train_loader):
#             states = states.to(device)
#             actions = actions.to(device)
#             rewards = rewards.to(device)
#             next_states = next_states.to(device)
#             dones = dones.to(device)

#             probs = actor(states)

#             q_values = critic(states, actions).squeeze()
#             with torch.no_grad():
#                 next_actions = target_actor(next_states)
#                 next_q_values = target_critic(next_states, next_actions).squeeze()
#                 target_q_values = rewards + gamma * next_q_values * (1 - dones)
            
#             critic_loss = nn.MSELoss()(q_values, target_q_values)
#             numerator = torch.sum((q_values - target_q_values) ** 2)
#             denominator = torch.sum(target_q_values ** 2) + 1e-8
#             nloss = numerator / denominator
#             critic_optimizer.zero_grad()
#             critic_loss.backward()
#             critic_optimizer.step()
#             #print(nloss)
#             critic_loss = nloss
            
#             # Debugging: Print critic loss
#             #print(f"Critic Loss: {critic_loss.item():.6f}")
            
#             # Update Actor
#             predicted_actions = actor(states)
#             #q_values = self.critic(states, predicted_actions)  # must be differentiable
#             #advantage = (q_values - q_values.mean()) / (q_values.std() + 1e-5)
#             #actor_loss = -advantage.mean()
            
#             beta = 0.01  # fixed entropy weight
#             q_values = critic(states, predicted_actions)
#             entropy = -(predicted_actions * torch.log(predicted_actions + 1e-8)).sum(dim=1).mean()
#             actor_loss = -q_values.mean() - beta * entropy

#             # warmup_steps = 100
#             # if iter > warmup_steps:
#             actor_optimizer.zero_grad()
#             actor_loss.backward()
#             actor_optimizer.step()
#             #for name, param in self.actor.named_parameters():
#             #    if param.grad is not None:
#             #        print(f"{name}: grad norm = {param.grad.norm().item()}")

#             #grad_norm = 0
#             #for p in self.actor.parameters():
#             #    if p.grad is not None:
#             #        grad_norm += p.grad.data.norm(2).item() ** 2
#             #print("Actor Gradient Norm:", grad_norm ** 0.5)

#             #print("Q values:", q_values[:5].tolist())
#             #print("Q mean:", q_values.mean().item())
#             #print("Q std dev:", q_values.std().item())

            
#             # Debugging: Print actor loss
#             #print(f"Actor Loss: {actor_loss.item():.10f}")
            
#             # Soft update target networks
#             soft_update(target_actor, actor, tau)
#             soft_update(target_critic, critic, tau)
#             #print(critic_loss.item(), actor_loss.item(), entropy.item())

#             actor_loss_list.append(actor_loss.item())
#             critic_loss_list.append(critic_loss.item())

#         actor_training_loss_history.append(np.mean(actor_loss_list))
#         critic_training_loss_history.append(np.mean(critic_loss_list))
#         reward_history.append(avg_reward)

#         update_training_results(
#             epoch=epoch,
#             actor_loss=actor_training_loss_history,
#             critic_loss=critic_training_loss_history,
#             reward=reward_history,
#         )

#         # Validation phase
#         # actor.eval()
#         # critic.eval()
#         # target_actor.eval()
#         # target_critic.eval()
#         # actor_val_loss_list = []
#         # critic_val_loss_list = []

#         # for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(valid_loader):
#         #     states = states.to(device)
#         #     actions = actions.to(device)
#         #     rewards = rewards.to(device)
#         #     next_states = next_states.to(device)
#         #     dones = dones.to(device)

#         #     probs = actor(states)

#         #     q_values = critic(states, actions).squeeze()
#         #     with torch.no_grad():
#         #         next_actions = target_actor(next_states)
#         #         next_q_values = target_critic(next_states, next_actions).squeeze()
#         #         target_q_values = rewards + gamma * next_q_values * (1 - dones)
            
#         #     critic_loss = nn.MSELoss()(q_values, target_q_values)
#         #     numerator = torch.sum((q_values - target_q_values) ** 2)
#         #     denominator = torch.sum(target_q_values ** 2) + 1e-8
#         #     nloss = numerator / denominator
#         #     #print(nloss)
#         #     critic_loss = nloss
            
#         #     # Debugging: Print critic loss
#         #     #print(f"Critic Loss: {critic_loss.item():.6f}")
            
#         #     # Update Actor
#         #     predicted_actions = actor(states)
#         #     #q_values = self.critic(states, predicted_actions)  # must be differentiable
#         #     #advantage = (q_values - q_values.mean()) / (q_values.std() + 1e-5)
#         #     #actor_loss = -advantage.mean()
            
#         #     beta = 0.01  # fixed entropy weight
#         #     q_values = critic(states, predicted_actions)
#         #     entropy = -(predicted_actions * torch.log(predicted_actions + 1e-8)).sum(dim=1).mean()
#         #     actor_loss = -q_values.mean() - beta * entropy
#         #     #grad_norm = 0
#         #     #for p in self.actor.parameters():
#         #     #    if p.grad is not None:
#         #     #        grad_norm += p.grad.data.norm(2).item() ** 2
#         #     #print("Actor Gradient Norm:", grad_norm ** 0.5)

#         #     #print("Q values:", q_values[:5].tolist())
#         #     #print("Q mean:", q_values.mean().item())
#         #     #print("Q std dev:", q_values.std().item())

            
#         #     # Debugging: Print actor loss
#         #     #print(f"Actor Loss: {actor_loss.item():.10f}")

#         #     actor_val_loss_list.append(actor_loss.item())
#         #     critic_val_loss_list.append(critic_loss.item())

#         # actor_validation_loss_history.append(np.mean(actor_val_loss_list))
#         # critic_validation_loss_history.append(np.mean(critic_val_loss_list))

#         print(f"Epoch {epoch}/{epochs}")
#         print(f"  Avg Reward: {avg_reward:.5f}")
#         print(f"  Train - Actor: {np.mean(actor_loss_list):.5f}, Critic: {np.mean(critic_loss_list):.5f}")
#         # print(f"  Valid - Actor: {np.mean(actor_val_loss_list):.5f}, Critic: {np.mean(critic_val_loss_list):.5f}")
#         print("========================================================")
        
#         # Save model checkpoint at local
#         if epoch % 10 == 0 or epoch == epochs - 1:
#             # torch.save(actor.state_dict(), f'./models/test_actor_epoch_{epoch}.pt')
#             # torch.save(critic.state_dict(), f'./models/test_critic_epoch_{epoch}.pt')
#             # print(f"Model saved at epoch {epoch}")
            
#             # Plot losses
#             plt.figure(figsize=(12, 4))
            
#             plt.subplot(1, 3, 1)
#             plt.plot(actor_training_loss_history, label='Training')
#             # plt.plot(actor_validation_loss_history, label='Validation')
#             plt.title('Policy Loss')
#             plt.xlabel('Epochs')
#             plt.ylabel('Loss')
#             plt.legend()
#             plt.grid()
            
#             plt.subplot(1, 3, 2)
#             plt.semilogy(critic_training_loss_history, label='Training')
#             # plt.semilogy(critic_validation_loss_history, label='Validation')
#             plt.title('Value Loss')
#             plt.xlabel('Epochs')
#             plt.ylabel('Loss')
#             plt.legend()
#             plt.grid()

#             plt.subplot(1, 3, 3)
#             plt.plot(reward_history, label='Average Reward', color='green')
#             plt.title('Average Reward per Epoch')
#             plt.xlabel('Epochs')        
#             plt.ylabel('Average Reward')
#             plt.legend()
#             plt.grid()
            
#             plt.tight_layout()
#             plt.savefig(f'./results/training_curve.png')
#             plt.close()

#     return actor_training_loss_history, critic_training_loss_history, reward_history, len(train_dataset)

def pretrain_fn(actor, critic, train_dataset, valid_dataset, batch_size=32, epochs=10, lr=1e-4, gamma=0, tau=0.1, device='cpu'):
    # Initialize models and optimizers
    actor.to(device)
    critic.to(device)
    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator = torch.Generator(device=device),
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator = torch.Generator(device=device),
    )

    # Create lists to store loss history
    actor_training_loss_history = []
    critic_training_loss_history = []
    reward_history = []

    for epoch in range(epochs):
        # Training phase
        actor.train()
        critic.train()
        target_actor.train()
        target_critic.train()
        actor_loss_list = []
        critic_loss_list = []
        reward_list = []
        
        for batch_idx, (states, actions, rewards, next_states) in enumerate(train_loader):
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)

            probs = actor(states)
            q_values = critic(states, actions).squeeze()
            with torch.no_grad():
                next_actions = target_actor(next_states)
                next_q_values = target_critic(next_states, next_actions).squeeze()
                # target_q_values = rewards + gamma * next_q_values * (1 - dones)
                target_q_values = rewards + gamma * next_q_values
            
            critic_loss = nn.MSELoss()(q_values, target_q_values)
            numerator = torch.sum((q_values - target_q_values) ** 2)
            denominator = torch.sum(target_q_values ** 2) + 1e-8
            nloss = numerator / denominator
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            #print(nloss)
            critic_loss = nloss
            
            # Debugging: Print critic loss
            #print(f"Critic Loss: {critic_loss.item():.6f}")
            
            # Update Actor
            predicted_actions = actor(states)
            #q_values = self.critic(states, predicted_actions)  # must be differentiable
            #advantage = (q_values - q_values.mean()) / (q_values.std() + 1e-5)
            #actor_loss = -advantage.mean()
            
            beta = 0.01  # fixed entropy weight
            q_values = critic(states, predicted_actions)
            entropy = -(predicted_actions * torch.log(predicted_actions + 1e-8)).sum(dim=1).mean()
            actor_loss = -q_values.mean() - beta * entropy

            # warmup_steps = 100
            # if iter > warmup_steps:
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            #for name, param in self.actor.named_parameters():
            #    if param.grad is not None:
            #        print(f"{name}: grad norm = {param.grad.norm().item()}")

            #grad_norm = 0
            #for p in self.actor.parameters():
            #    if p.grad is not None:
            #        grad_norm += p.grad.data.norm(2).item() ** 2
            #print("Actor Gradient Norm:", grad_norm ** 0.5)

            #print("Q values:", q_values[:5].tolist())
            #print("Q mean:", q_values.mean().item())
            #print("Q std dev:", q_values.std().item())

            
            # Debugging: Print actor loss
            #print(f"Actor Loss: {actor_loss.item():.10f}")
            
            # Soft update target networks
            soft_update(target_actor, actor, tau)
            soft_update(target_critic, critic, tau)
            #print(critic_loss.item(), actor_loss.item(), entropy.item())

            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            reward_list.append(0)

        actor_training_loss_history.append(np.mean(actor_loss_list))
        critic_training_loss_history.append(np.mean(critic_loss_list))
        reward_history.append(np.mean(reward_list))

        # Update training results to MongoDB
        update_training_results(
            epoch=epoch,
            actor_loss=actor_training_loss_history,
            critic_loss=critic_training_loss_history,   
            reward = reward_history)

        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train - Actor: {np.mean(actor_loss_list):.5f}, Critic: {np.mean(critic_loss_list):.5f}, Reward: {np.mean(reward_list):.5f}")
        print("========================================================")
        
        # Save model checkpoint at local
        # if epoch % 10 == 0 or epoch == epochs - 1:
        #     # Plot losses
        #     plt.figure(figsize=(12, 4))
            
        #     plt.subplot(1, 2, 1)
        #     plt.plot(actor_training_loss_history, label='Training')
        #     plt.title('Policy Loss')
        #     plt.xlabel('Epochs')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     plt.grid()
            
        #     plt.subplot(1, 2, 2)
        #     plt.semilogy(critic_training_loss_history, label='Training')
        #     plt.title('Value Loss')
        #     plt.xlabel('Epochs')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     plt.grid()
            
        #     plt.tight_layout()
        #     plt.savefig(f'./results/training_loss.png')
        #     plt.close()
    
    return np.mean(actor_training_loss_history), np.mean(critic_training_loss_history), np.mean(reward_history)

def update_training_results(epoch: int,
                            actor_loss: list,
                            critic_loss: list,
                            reward: list,):
    # Connect to MongoDB and load training parameters
    mongodb_url = os.environ.get("AITRCOMMONDB_URI")
    client = MongoClient(mongodb_url)
    database = client["TrainingConfig"]
    collection = database["current_task"]

    # Query data from MongoDB
    item = collection.find_one({
        "status": "running"
    })

    project_id = item["project_id"]
    app_name = item["app_name"]
    model_name = item["model_name"]
    model_version = item["model_version"]
    mode = item["mode"]
    dataset_name = item["dataset_name"]
    total_epochs = item["epochs"]
    learning_rate = item["learning_rate"]

    # Connect to MongoDB and save training results
    mongodb_url = os.environ.get("AITRCOMMONDB_URI")
    client = MongoClient(mongodb_url)
    database = client["TrainingResults"]
    collection_name = f"{project_id}_{app_name}_{model_name}_{model_version}_{mode}_{dataset_name}"
    collection = database[collection_name]

    filter_data = {
        "project_id": project_id,
        "app_name": app_name,
        "model_name": model_name,
        "model_version": model_version,
        "mode": mode,
        "dataset_name": dataset_name,
    }

    record = {
        "total_epochs": total_epochs,
        "current_epoch": epoch,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "reward": reward,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    collection.update_one(filter_data, {"$set": record}, upsert=True)