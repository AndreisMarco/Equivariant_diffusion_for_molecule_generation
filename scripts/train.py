import os
import time
import json
from tqdm import tqdm
import yaml
from collections import OrderedDict

import numpy as np
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader

from scripts.process_QM9 import load_QM9_split
from scripts.utils import  FullyConnectedEdgeIndex, SizeDistribution, save_and_plot_loss, write_xyz
from scripts.models.DDPM import DDPM
from scripts.models.EGNN_Satorras import EGNN as EGNN_Satorras
from scripts.modells.EGNN_Hoogeboom import EGNN as EGNN_Hoogeboom
    
def train(model, optimizer, scheduler, data_loader, 
          num_epochs, ckpt_epochs, save_ckpt, run_path,
          start_epoch, cum_time, loss_history, device):
    model.train()
    # Initialize progress bar and loss bookeeping
    progress_bar = tqdm(range(start_epoch, num_epochs), desc="Training", unit="Epochs")
    if start_epoch == 0:
        loss_history = {"xh": [], "x": [], "h": []}
    else:
        loss_history = loss_history

    print("Starting training")
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        cum_loss = {"xh": 0.0, "x": 0.0, "h": 0.0}
        for batched_data in data_loader:
            # Transfer batch to GPU
            batched_data = batched_data.to(device)
            optimizer.zero_grad()
            # Compute loss and gradients
            loss = model.loss(batched_data)
            loss["xh"].backward()
            # Update parameters
            optimizer.step()
            # Update cumulative losses
            for key in cum_loss.keys(): 
                cum_loss[key] += loss[key].item()
        # Update learning rate
        scheduler.step()

        # Log the losses and update progress bar
        for key in loss_history.keys():
            loss_history[key].append(cum_loss[key] / len(data_loader))
        progress_bar.set_postfix(OrderedDict([
                                          ('loss', f"{loss_history['xh'][-1]:.4f}"),
                                          ('x_loss', f"{loss_history['x'][-1]:.4f}"),
                                          ('h_loss', f"{loss_history['h'][-1]:.4f}")
                                          ]))
        progress_bar.update()
        # Save checkpoint if reached specified epoch or last epoch
        is_periodic_save = (epoch + 1) % ckpt_epochs == 0
        is_last_epoch = (epoch + 1) == num_epochs
        if save_ckpt and (is_periodic_save or is_last_epoch):
            current_session_time = time.time() - start_time
            cum_time += current_session_time
            start_time = time.time()
            ckpt_single_path = os.path.join(run_path, "checkpoints", f"ckpt_{epoch}.pt")
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "cum_time": cum_time
            }
            torch.save(obj=checkpoint, f=ckpt_single_path)
            save_and_plot_loss(loss_history=loss_history, save_path=run_path)
    
    current_session_time = time.time() - start_time
    cum_time += current_session_time
    print(f"Finished training! Training time: {cum_time} s")
    return cum_time

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/cfg.yaml', help='Path to yaml configuration file (default: %(default)s)')
    args = parser.parse_args()

    cfg_path = args.cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Set seed for reproducibility
    set_seed(cfg["seed"])
    # Load dataset
    if cfg["train_on_subset"]:
        train_set, train_infos, _ = load_QM9_split(data_dir=cfg["data_dir"],
                                                split="train_subset",
                                                load_info=True)
    else:
        train_set, train_infos, _ = load_QM9_split(data_dir=cfg["data_dir"],
                                                split="train",
                                                load_info=True)
    print(f"Loaded training set (total datapoints: {len(train_set)})")
    
    # Istantiate model
    mol_size_hist = {int(size): count for size, count in train_infos["mol_size_hist"].items()}
    graph_size_dist = SizeDistribution(sizes=list(mol_size_hist.keys()),
                                       counts=list(mol_size_hist.values()))
    edge_index_builder = FullyConnectedEdgeIndex(graphs_sizes=list(mol_size_hist.keys()))
    
    model_dict = {"Satorras":EGNN_Satorras,
                  "Hoogeboom": EGNN_Hoogeboom}
    
    noise_net = model_dict[cfg["noise_net"]["type"]]
    noise_net = noise_net(input_dim=cfg["noise_net"]["input_dim"],
                          hidden_dim=cfg["noise_net"]["hidden_dim"],
                          num_layers=cfg["noise_net"]["num_layers"],
                          agg_method=cfg["noise_net"]["agg_method"],
                          norm_factor=cfg["noise_net"]["norm_factor"],
                          # Satorras exclusive
                          d_proj_dim=cfg["noise_net"]["d_proj_dim"],
                          t_emb_dim=cfg["noise_net"]["t_emb_dim"],
                          # Hoogeboom exclusive
                          num_sublayers=cfg["noise_net"]["num_sublayers"],
                          tanh=cfg["noise_net"]["tanh"],
                          coords_range=cfg["noise_net"]["coords_range"],
                          sin_embedding=cfg["noise_net"]["sin_embedding"]
                          ).to(cfg["device"])

    model = DDPM(noise_net=noise_net,
                 T=cfg["ddpm"]["T"],
                 beta_min=cfg["ddpm"]["beta_min"],
                 beta_max=cfg["ddpm"]["beta_max"],
                 input_dim=cfg["ddpm"]["input_dim"],
                 coord_dim=cfg["ddpm"]["coord_dim"],
                 loss_weights=cfg["train"]["loss_weights"],
                 edge_index_builder=edge_index_builder,
                 graph_size_dist=graph_size_dist).to(cfg["device"])
    print("Istantiated model")
    if cfg["mode"] == "train":
        # Training setup
        start_epoch = 0
        g = torch.Generator()
        g.manual_seed(cfg["seed"]) 
        train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, generator=g)
        lr = cfg["train"]["lr"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        num_epochs = cfg["train"]["num_epochs"]
        # Learning rate schedule = linear warmp up + cosine decay
        warm_up_epochs = cfg["train"]["warmup_epochs"]
        warm_up_scheduler = LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warm_up_epochs)
        cosine_annealing_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warm_up_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warm_up_scheduler, cosine_annealing_scheduler], milestones=[warm_up_epochs])
        
        run_path = cfg["run_path"]
        if not cfg["train"]["resume_from_ckpt"]:
            # Assure existance of a new logging path
            i = 0
            while os.path.exists(run_path):
                print(f"Provided run path {run_path} already exists")
                run_path = f"{cfg['run_path']}_{i}"
                i += 1
            # Create the new directory
            print(f"Saving logs and model at {run_path}")
            os.makedirs(run_path)
            cfg["run_path"] = run_path
            # If ckpt_epochs is specified create a folder to save ckpts to
            save_ckpt = False
            if cfg["train"]["ckpt_epochs"] is not None:
                save_ckpt = True
                ckpt_path = os.path.join(cfg["run_path"], "checkpoints/")
                os.makedirs(ckpt_path)
            loss_history = None
            cum_time = 0.0
        else: 
            # Assure existance of model source
            if not os.path.exists(cfg["run_path"]):
                print(f"The run path {cfg['run_path']} does not exist")
                exit()
            # Check that there is at least one ckpt
            ckpt_dir = os.path.join(run_path, "checkpoints")
            if not os.listdir(ckpt_dir):
                print(f"No checkpoints found in {ckpt_dir}")
                exit()
            # Load train states from last checkpoint
            last_ckpt = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))[-1] 
            last_ckpt_file = os.path.join(ckpt_dir, last_ckpt)
            checkpoint = torch.load(last_ckpt_file, map_location=cfg["device"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            save_ckpt = True
            cum_time = checkpoint.get("cum_time", 0.0)
            # Load loss file
            loss_file = os.path.join(run_path, "loss.json")
            with open(loss_file, "r") as f:
                loss_history = json.load(f)
            print(f"Resuming training from checkpoint {last_ckpt}")

        # Train model 
        final_cum_time = train(model=model,
              optimizer=optimizer,
              scheduler=scheduler,
              data_loader=train_loader,
              num_epochs=num_epochs,
              ckpt_epochs=cfg["train"]["ckpt_epochs"],
              save_ckpt=save_ckpt,
              run_path=cfg["run_path"],
              start_epoch=start_epoch,
              cum_time=cum_time,
              loss_history=loss_history,
              device=cfg["device"])
        
        # Save final model
        model_path = os.path.join(cfg["run_path"], "model.pt")
        final = {"epoch": num_epochs,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "cum_time": final_cum_time}
        torch.save(obj=final, f=model_path)
        # Save a copy of the training configurations
        # Include total training time
        cfg["total_training_time_seconds"] = final_cum_time
        # Include total learnable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Learnable Parameters: {total_params}")
        cfg["model_parameters"] = total_params
        cfg_copy_path = os.path.join(run_path, "cfg.yaml")
        with open(cfg_copy_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        
        if cfg["sample_after_train"]:
            cfg["mode"] = "sample"

    if cfg["mode"] == "sample":
        # Assure existance of model source
        if not os.path.exists(cfg["run_path"]):
            print(f"The run path {cfg['run_path']} does not exist")
            exit()
        # Load model parameters
        model_path = os.path.join(cfg["run_path"], "model.pt")
        state = torch.load(model_path, map_location=cfg["device"])
        model.load_state_dict(state["model_state"])
        print(f"Loaded model from {model_path}")
        # Generate samples
        num_samples = cfg["sample"]["num_samples"]
        atom_decoder = np.array(cfg["sample"]["atom_decoder"]) 
        for i in range(3):
            x, h, batch = model.sample(num_samples=num_samples,
                                    batch_size=cfg["sample"]["batch_size"])
            # Convert h to atom type
            atom_decoder = np.array(cfg["sample"]["atom_decoder"])
            atom_type = h.argmax(dim=1).numpy()
            atom_type = atom_decoder[atom_type]
            # Write samples to xyz format
            sample_path = os.path.join(cfg["run_path"], f"samples{i}.xyz")
            write_xyz(atom_type=atom_type, coord=x, batch=batch, file=sample_path)
            print(f"Samples save at {sample_path}")
