import sys, os
import numpy as np
import torch
import torch.nn as nn

from netdt.metric_report import update_training_results_SL
from netdt.data_preprocessing import data_preprocessing

def train_fn(model, local_file_path, batch_size=32, epochs=10, lr=1e-4, device='cpu'):
    # Preprocessing data
    x_sim, y_sim_n, x_sim_n, x_meas, y_meas_n, x_meas_n, x_mu, x_std, y_meas_mu, y_meas_std, feature_names, label_name, ru_to_pci = data_preprocessing(local_file_path)
    x_sim_t = torch.from_numpy(x_sim_n).to(device)
    y_sim_t = torch.from_numpy(y_sim_n).to(device)
    x_meas_t = torch.from_numpy(x_meas_n).to(device)
    y_meas_t = torch.from_numpy(y_meas_n).to(device)

    datasize = x_sim_t.shape[0] + x_meas_t.shape[0]

    # Set training parameters
    PRE_EPOCHS = int(epochs/2)
    PRE_BATCH = batch_size
    PRE_STEPS_PER_EPOCH = 30
    PRE_LR = lr

    FT_EPOCHS = epochs - PRE_EPOCHS
    FT_BATCH_SIZE = batch_size
    FT_STEPS_PER_EPOCH = 10
    FT_LR = lr / 10
    FT_WEIGHT_DECAY = 1e-5
    LAMBDA_ANCHOR = 0.3
    TAU = 6.0
    
    # Initialize models and optimizers
    model.to(device)
    loss_fn = nn.SmoothL1Loss(beta=3.0)

    opt_pre = torch.optim.Adam(model.parameters(), lr=PRE_LR, weight_decay=1e-6)
    n_sim = x_sim_t.shape[0]

    # Initialize the history lists
    training_loss_history_stage1 = []
    training_loss_history_stage2 = []

    update_training_results_SL(total_epochs_stage1=PRE_EPOCHS,
                               epoch_stage1=0,
                               training_loss_stage1=training_loss_history_stage1,
                               total_epochs_stage2=FT_EPOCHS,
                               epoch_stage2=0,
                               training_loss_stage2=training_loss_history_stage2)

    print(f"[INFO] Pretrain from NPZ: samples={n_sim}, epochs={PRE_EPOCHS}")
    for epoch in range(1, PRE_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(PRE_STEPS_PER_EPOCH):
            idx = torch.randint(0, n_sim, (min(PRE_BATCH, n_sim),), device=device)
            xb = x_sim_t[idx]
            yb = y_sim_t[idx]
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            opt_pre.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pre.step()
            loss_sum += float(loss.detach().cpu())


        training_loss_history_stage1.append(loss_sum / PRE_STEPS_PER_EPOCH)
        update_training_results_SL(total_epochs_stage1=PRE_EPOCHS,
                               epoch_stage1=epoch,
                               training_loss_stage1=training_loss_history_stage1,
                               total_epochs_stage2=FT_EPOCHS,
                               epoch_stage2=0,
                               training_loss_stage2=training_loss_history_stage2)
        if epoch % 20 == 0:
            print(f"[Pretrain] epoch={epoch:3d} loss={loss_sum/PRE_STEPS_PER_EPOCH:.6f}")

    for p in model.enc.parameters():
        p.requires_grad = False
    for p in model.dec.parameters():
        p.requires_grad = True

    opt_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)

    y_sim_anchor_meas = ((x_meas[:, 0:1] - y_meas_mu) / y_meas_std).astype(np.float32)
    y_sim_anchor_meas_t = torch.from_numpy(y_sim_anchor_meas).to(device)

    y_sim_anchor_sim = ((x_sim[:, 0:1] - y_meas_mu) / y_meas_std).astype(np.float32)
    y_sim_anchor_sim_t = torch.from_numpy(y_sim_anchor_sim).to(device)

    y_meas_mu_t = torch.from_numpy(y_meas_mu).to(device)
    y_meas_std_t = torch.from_numpy(y_meas_std).to(device)
    x_sim_ft_t = x_sim_t

    n_meas = x_meas_t.shape[0]
    n_sim_ft = x_sim_ft_t.shape[0]
    print(f"[INFO] Finetune from NPZ: measured_samples={n_meas}, epochs={FT_EPOCHS}")

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        loss_meas_sum = 0.0
        loss_anchor_sum = 0.0
        loss_sim_sum = 0.0
        for _ in range(FT_STEPS_PER_EPOCH):
            idx = torch.randint(0, n_meas, (min(FT_BATCH_SIZE, n_meas),), device=device)
            xb = x_meas_t[idx]
            yb = y_meas_t[idx]
            y_sim_b = y_sim_anchor_meas_t[idx]

            pred, _ = model(xb)
            loss_meas = loss_fn(pred, yb)
            loss_anchor = loss_fn(pred, y_sim_b)

            idx2 = torch.randint(0, n_sim_ft, (min(FT_BATCH_SIZE, n_sim_ft),), device=device)
            xb2 = x_sim_ft_t[idx2]
            y_sim2 = y_sim_anchor_sim_t[idx2]

            pred2, _ = model(xb2)
            loss_sim = loss_fn(pred2, y_sim2)

            meas_db = (yb * y_meas_std_t + y_meas_mu_t).squeeze(1)
            sim_db = (y_sim_b * y_meas_std_t + y_meas_mu_t).squeeze(1)
            delta = torch.abs(meas_db - sim_db)
            lambda_i = LAMBDA_ANCHOR * torch.exp(-delta / TAU)

            loss = loss_meas + (lambda_i.mean() * loss_anchor) + 0.5 * loss_sim

            opt_ft.zero_grad(set_to_none=True)
            loss.backward()
            opt_ft.step()

            loss_meas_sum += float(loss_meas.detach().cpu())
            loss_anchor_sum += float(loss_anchor.detach().cpu())
            loss_sim_sum += float(loss_sim.detach().cpu())

        loss_sum = loss_meas_sum + loss_anchor_sum + loss_sim_sum
        training_loss_history_stage2.append(loss_sum / FT_STEPS_PER_EPOCH)
        update_training_results_SL(total_epochs_stage1=PRE_EPOCHS,
                               epoch_stage1=PRE_EPOCHS,
                               training_loss_stage1=training_loss_history_stage1,
                               total_epochs_stage2=FT_EPOCHS,
                               epoch_stage2=epoch,
                               training_loss_stage2=training_loss_history_stage2)
        if epoch % 30 == 0:
            print(
                f"[Finetune] epoch={epoch:3d} "
                f"meas={loss_meas_sum/FT_STEPS_PER_EPOCH:.6f} "
                f"anchor={loss_anchor_sum/FT_STEPS_PER_EPOCH:.6f} "
                f"sim={loss_sim_sum/FT_STEPS_PER_EPOCH:.6f}"
            )
            
    return np.mean(training_loss_history_stage1), np.mean(training_loss_history_stage2), datasize

def test_fn(model, local_file_path, batch_size=32, epochs=10, lr=1e-4, device='cpu'):
    # Preprocessing data
    x_sim, y_sim_n, x_sim_n, x_meas, y_meas_n, x_meas_n, x_mu, x_std, y_meas_mu, y_meas_std, feature_names, label_name, ru_to_pci = data_preprocessing(local_file_path)
    x_sim_t = torch.from_numpy(x_sim_n).to(device)
    y_sim_t = torch.from_numpy(y_sim_n).to(device)
    x_meas_t = torch.from_numpy(x_meas_n).to(device)
    y_meas_t = torch.from_numpy(y_meas_n).to(device)

    datasize = x_sim_t.shape[0] + x_meas_t.shape[0]

    # Set training parameters
    PRE_EPOCHS = int(epochs/2)
    PRE_BATCH = batch_size
    PRE_STEPS_PER_EPOCH = 30
    PRE_LR = lr

    FT_EPOCHS = epochs - PRE_EPOCHS
    FT_BATCH_SIZE = batch_size
    FT_STEPS_PER_EPOCH = 10
    FT_LR = lr / 10
    FT_WEIGHT_DECAY = 1e-5
    LAMBDA_ANCHOR = 0.3
    TAU = 6.0
    
    # Initialize models and optimizers
    model.to(device)
    loss_fn = nn.SmoothL1Loss(beta=3.0)

    opt_pre = torch.optim.Adam(model.parameters(), lr=PRE_LR, weight_decay=1e-6)
    n_sim = x_sim_t.shape[0]

    # Initialize the history lists
    training_loss_history_stage1 = []
    training_loss_history_stage2 = []

    print(f"[INFO] Pretrain from NPZ: samples={n_sim}, epochs={PRE_EPOCHS}")
    for epoch in range(1, PRE_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(PRE_STEPS_PER_EPOCH):
            idx = torch.randint(0, n_sim, (min(PRE_BATCH, n_sim),), device=device)
            xb = x_sim_t[idx]
            yb = y_sim_t[idx]
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            opt_pre.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pre.step()
            loss_sum += float(loss.detach().cpu())


        training_loss_history_stage1.append(loss_sum / PRE_STEPS_PER_EPOCH)
        if epoch % 20 == 0:
            print(f"[Pretrain] epoch={epoch:3d} loss={loss_sum/PRE_STEPS_PER_EPOCH:.6f}")

    for p in model.enc.parameters():
        p.requires_grad = False
    for p in model.dec.parameters():
        p.requires_grad = True

    opt_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)

    y_sim_anchor_meas = ((x_meas[:, 0:1] - y_meas_mu) / y_meas_std).astype(np.float32)
    y_sim_anchor_meas_t = torch.from_numpy(y_sim_anchor_meas).to(device)

    y_sim_anchor_sim = ((x_sim[:, 0:1] - y_meas_mu) / y_meas_std).astype(np.float32)
    y_sim_anchor_sim_t = torch.from_numpy(y_sim_anchor_sim).to(device)

    y_meas_mu_t = torch.from_numpy(y_meas_mu).to(device)
    y_meas_std_t = torch.from_numpy(y_meas_std).to(device)
    x_sim_ft_t = x_sim_t

    n_meas = x_meas_t.shape[0]
    n_sim_ft = x_sim_ft_t.shape[0]
    print(f"[INFO] Finetune from NPZ: measured_samples={n_meas}, epochs={FT_EPOCHS}")

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        loss_meas_sum = 0.0
        loss_anchor_sum = 0.0
        loss_sim_sum = 0.0
        for _ in range(FT_STEPS_PER_EPOCH):
            idx = torch.randint(0, n_meas, (min(FT_BATCH_SIZE, n_meas),), device=device)
            xb = x_meas_t[idx]
            yb = y_meas_t[idx]
            y_sim_b = y_sim_anchor_meas_t[idx]

            pred, _ = model(xb)
            loss_meas = loss_fn(pred, yb)
            loss_anchor = loss_fn(pred, y_sim_b)

            idx2 = torch.randint(0, n_sim_ft, (min(FT_BATCH_SIZE, n_sim_ft),), device=device)
            xb2 = x_sim_ft_t[idx2]
            y_sim2 = y_sim_anchor_sim_t[idx2]

            pred2, _ = model(xb2)
            loss_sim = loss_fn(pred2, y_sim2)

            meas_db = (yb * y_meas_std_t + y_meas_mu_t).squeeze(1)
            sim_db = (y_sim_b * y_meas_std_t + y_meas_mu_t).squeeze(1)
            delta = torch.abs(meas_db - sim_db)
            lambda_i = LAMBDA_ANCHOR * torch.exp(-delta / TAU)

            loss = loss_meas + (lambda_i.mean() * loss_anchor) + 0.5 * loss_sim

            opt_ft.zero_grad(set_to_none=True)
            loss.backward()
            opt_ft.step()

            loss_meas_sum += float(loss_meas.detach().cpu())
            loss_anchor_sum += float(loss_anchor.detach().cpu())
            loss_sim_sum += float(loss_sim.detach().cpu())

        loss_sum = loss_meas_sum + loss_anchor_sum + loss_sim_sum
        training_loss_history_stage2.append(loss_sum / FT_STEPS_PER_EPOCH)
        if epoch % 30 == 0:
            print(
                f"[Finetune] epoch={epoch:3d} "
                f"meas={loss_meas_sum/FT_STEPS_PER_EPOCH:.6f} "
                f"anchor={loss_anchor_sum/FT_STEPS_PER_EPOCH:.6f} "
                f"sim={loss_sim_sum/FT_STEPS_PER_EPOCH:.6f}"
            )
            
    return np.mean(training_loss_history_stage1), np.mean(training_loss_history_stage2), datasize