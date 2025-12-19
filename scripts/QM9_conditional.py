import os
import json
from tqdm import tqdm
import yaml
from collections import OrderedDict

import numpy as np
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as GraphObject 

from scripts.process_QM9 import load_QM9_split
from scripts.utils import center_positions, FullyConnectedEdgeIndex, SizeDistribution, SinusoidalEmbedding, save_and_plot_loss

class EGNNIsoform(nn.Module):
    def __init__(self,
                atom_pos_dim=3, 
                atom_feat_dim=5,
                t_dim=6,
                h_proj_dim=128,
                t_proj_dim=128,
                d_proj_dim=32,
                message_dim=32,
                num_mp_rounds=3, 
                activation_fn=nn.SiLU, 
                conditioning="cat",
                normalize_aggr=True, 
                norm_factor=100):
        super().__init__()
        self.atom_pos_dim = atom_pos_dim
        self.atom_feat_dim = atom_feat_dim
        self.message_dim = message_dim
        self.num_mp_rounds = num_mp_rounds
        self.conditioning = conditioning
        self.normalize_aggr = normalize_aggr
        self.norm_factor = norm_factor

        if self.conditioning == "cat":
            message_input_dim = h_proj_dim * 3 + t_proj_dim + d_proj_dim
        elif self.conditioning == "sum":
            assert t_proj_dim == h_proj_dim, \
                f"For conditioning by adding the timestep t_proj_dim == h_proj_dim, got {t_proj_dim=} and {h_proj_dim=}"
            message_input_dim = h_proj_dim * 2 + d_proj_dim

        self.h_projection_net = nn.Sequential(
            nn.Linear(self.atom_feat_dim, h_proj_dim),
            activation_fn(),
            nn.Linear(h_proj_dim, h_proj_dim))

        self.dist_projection_net = nn.Sequential(
            nn.Linear(1, d_proj_dim),
            activation_fn(),
            nn.Linear(d_proj_dim, d_proj_dim))

        self.t_projection_net = nn.Sequential(
            nn.Linear(t_dim, t_proj_dim),
            activation_fn(),
            nn.Linear(t_proj_dim, t_proj_dim))
        
        self.c_projection_net = nn.Sequential(
            nn.Linear(self.atom_feat_dim, h_proj_dim),
            activation_fn(),
            nn.Linear(h_proj_dim, h_proj_dim))
            
        self.message_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(message_input_dim, message_dim),
                activation_fn(),
                nn.Linear(message_dim, message_dim),
                activation_fn(),
                nn.Linear(message_dim, message_dim),
               activation_fn(),
                nn.Linear(message_dim, h_proj_dim)
            ) for r in range(num_mp_rounds)
        ])

        self.scalar_weight_net = nn.Sequential(
            nn.Linear(h_proj_dim, h_proj_dim),
            activation_fn(),
            nn.Linear(h_proj_dim, 1),
            nn.Tanh())
        
        self.out_network = nn.Linear(h_proj_dim, atom_feat_dim)

    def forward(self, batched_graphs, t):
        x = batched_graphs.x
        h = batched_graphs.h
        c = batched_graphs.c
        batch = batched_graphs.batch
        row, col = batched_graphs.edge_index

        norm_factor = 1.0
        if self.normalize_aggr:    
            if self.norm_factor == "neighbours":
                norm_factor = torch.bincount(row).unsqueeze(1).float() 
            else:
                norm_factor = self.norm_factor

        h = self.h_projection_net(h)
        t = self.t_projection_net(t)
        c = self.c_projection_net(c)
        for r in range(self.num_mp_rounds):
            sq_dist = ((x[row] - x[col]) ** 2).sum(dim=1, keepdim=True)
            dist = torch.sqrt(torch.clamp(sq_dist, min=1e-12))
            d = self.dist_projection_net(dist)
            if self.conditioning == "cat":
                message_net_input = torch.cat([h[row], h[col], t[batch[row], c[batch[row]]], d], dim=1)
            elif self.conditioning == "sum": 
                message_net_input = torch.cat([h[row] + t[batch[row]] + c[batch[row]],
                                               h[col] + t[batch[col]] + c[batch[col]], d], dim=1)
            messages = self.message_net[r](message_net_input) 

            # position update 
            scalar_weights = self.scalar_weight_net(messages)
            pos_updates = (x[row] - x[col])*scalar_weights 
            aggr_pos_updates = torch.zeros_like(x)  
            aggr_pos_updates = aggr_pos_updates.index_add_(0, row, pos_updates) 
            aggr_pos_updates = aggr_pos_updates / norm_factor
            x = x + aggr_pos_updates 

            # feature update
            aggr_messages = torch.zeros_like(h)
            aggr_messages = aggr_messages.index_add_(0, row, messages) 
            aggr_messages = aggr_messages / norm_factor
            h = h + aggr_messages
        h = self.out_network(h)
        return x, h

class DDPM(nn.Module):
    '''
    Implements the DDPM introducted by Ho et al. (2020) adapted for molecular data
    '''
    def __init__(self, 
                 beta_min, 
                 beta_max, 
                 T,
                 noise_network, 
                 edge_index_builder, 
                 graph_size_dist,
                 atom_pos_dim, 
                 atom_feat_dim, 
                 t_dim,
                 loss_weights):
        super(DDPM, self).__init__()
        self.T = T
        self.register_buffer('beta', torch.linspace(beta_min, beta_max, T))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', self.alpha.cumprod(dim=0))

        self.atom_pos_dim = atom_pos_dim
        self.atom_feat_dim = atom_feat_dim
        self.t_dim = t_dim
        self.edge_index_builder = edge_index_builder
        self.graph_size_dist = graph_size_dist
        self.t_embedding = SinusoidalEmbedding(norm_factor=10000, embedding_dim=t_dim)
        self.noise_network = noise_network
        self.loss_weights = loss_weights

    def loss(self, batched_graphs):
        batch = batched_graphs.batch
        num_graphs = int(batch.max().item()) + 1
        c = torch.zeros(size=(num_graphs, batched_graphs.h.shape[1]), device=batch.device)
        c = c.index_add_(0, batch, batched_graphs.h)
        # sample timesteps
        t = torch.randint(0, self.T, (num_graphs,)).to(batch.device)
        t = t.view(-1, 1)
        t_batch = t[batch]
        # sample epsilon
        eps_x = torch.randn_like(batched_graphs.x)
        eps_x = center_positions(eps_x, batch)
        eps_h = torch.randn_like(batched_graphs.h)
        # compute noised z_t
        alpha_bar_t = self.alpha_bar[t_batch]
        x_t = torch.sqrt(alpha_bar_t) * batched_graphs.x + torch.sqrt(1 - alpha_bar_t) * eps_x
        h_t = torch.sqrt(alpha_bar_t) * batched_graphs.h + torch.sqrt(1 - alpha_bar_t) * eps_h
        # predict noise
        t_emb = self.t_embedding(t)
        fc_edge_index = self.edge_index_builder(batch)
        noised_graphs = GraphObject(x=x_t,
                                    h=h_t,
                                    batch=batch,
                                    edge_index=fc_edge_index,
                                    c=c)
        eps_theta_x, eps_theta_h = self.noise_network(noised_graphs, t_emb)
        # compute MSE loss 
        mse_loss_x = ((eps_x - eps_theta_x)**2)
        mse_loss_x = mse_loss_x.mean() 
        mse_loss_h = ((eps_h - eps_theta_h)**2)
        mse_loss_h = mse_loss_h.mean() 
        loss_dict = {"xh": mse_loss_x * self.loss_weights["x"] + mse_loss_h * self.loss_weights["h"],
                     "x": mse_loss_x,
                     "h": mse_loss_h}
        return loss_dict
    
    @torch.no_grad()
    def sample(self, num_samples, batch_size, c, device):
        if c is None:
            # Sample n_samples sizes from the empirical size distribution of the training set
            num_nodes = self.graph_size_dist.sample(num_samples)
        else:
            # The number of atoms is given by the specified isoform 
            assert len(c) == 5, "Invalid isoform"
            c = torch.tensor(c)
            num_nodes = torch.full(size=(num_samples,), fill_value=c.sum())
        all_x, all_h, all_batch = [], [], []
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Sampling"):
            sizes_curr = num_nodes[i:i + batch_size]
            n_curr = int(len(sizes_curr))
            total_nodes_curr = sum(sizes_curr)
            # Create data.batch-like tensor and transform to fc_index_list 
            batch = torch.repeat_interleave(torch.arange(n_curr), sizes_curr).to(device)
            if c is not None:
                c_batch = c.unsqueeze(0).repeat(n_curr, 1).float().to(device=device)
            else:
                c_batch = None
            fc_edge_index = self.edge_index_builder(batch)
            # Sample zero CoM node pos and feats from normal prior
            x_t = torch.randn((total_nodes_curr, self.atom_pos_dim), device=device)
            x_t = center_positions(x_t, batch)
            h_t = torch.randn((total_nodes_curr, self.atom_feat_dim), device=device)
            # Reverse diffusion
            for t in range(self.T - 1, -1, -1):
                t_graphs = torch.full((n_curr, 1), t, device=device)
                if t > 0:
                    sigma = torch.sqrt(self.beta[t])
                    x_z = torch.randn_like(x_t)
                    h_z = torch.randn_like(h_t)
                else:
                    sigma = 0.0
                    x_z = torch.zeros_like(x_t)
                    h_z = torch.zeros_like(h_t)
                # Predict current step noise
                t_emb = self.t_embedding(t_graphs)
                noised_graphs = GraphObject(x=x_t,
                                        h=h_t,
                                        batch=batch,
                                        edge_index=fc_edge_index,
                                        c=c_batch)
                eps_theta_x, eps_theta_h = self.noise_network(noised_graphs, t_emb)
                # Compute one-step denoised latent
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                x_t = (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + 1e-8)) * eps_theta_x) \
                    / torch.sqrt(alpha_t + 1e-8) + sigma * x_z
                x_t = center_positions(pos=x_t, batch=batch)
                h_t = (h_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + 1e-8)) * eps_theta_h) \
                    / torch.sqrt(alpha_t + 1e-8) + sigma * h_z
            all_x.append(x_t)
            all_h.append(h_t)
            all_batch.append(batch + i)  # adjust batch values
        all_x = torch.cat(all_x).cpu()
        all_h = torch.cat(all_h).cpu()
        all_batch = torch.cat(all_batch).cpu()
        return all_x, all_h, all_batch
    

def train(model, optimizer, scheduler, data_loader, 
          num_epochs, ckpt_epochs, save_ckpt, run_path,
          start_epoch, loss_history, device):
    model.train()
    # Initialize progress bar and loss bookeeping
    progress_bar = tqdm(range(start_epoch, num_epochs), desc="Training", unit="Epochs")
    if start_epoch == 0:
        loss_history = {"xh": [], "x": [], "h": []}
    else:
        loss_history = loss_history

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
        if save_ckpt and (epoch+1) % ckpt_epochs == 0 and epoch != 0:
            ckpt_single_path = os.path.join(run_path, "checkpoints", f"ckpt_{epoch}.pt")
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, ckpt_single_path)
            save_and_plot_loss(loss_history=loss_history, save_path=run_path)

@torch.no_grad()
def test(model, data_loader, device):
    model.eval()
    # Initialize progress bar
    num_batches = len(data_loader)
    progress_bar = tqdm(range(num_batches), desc="Testing", unit="Epochs")
    cum_loss = {"xh": 0.0, "x": 0.0, "h": 0.0}   
    for batched_graphs in data_loader: 
        # Transfer batch to GPU
        batched_graphs = batched_graphs.to(device) 
         # Compute loss and update comulative losses
        loss = model.loss(batched_graphs)
        for key in cum_loss.keys(): 
                cum_loss[key] += loss[key].item()

        # Update progress bar
        avg_losses = {key: (cum_loss[key] / num_batches) for key in cum_loss}
        progress_bar.set_postfix(OrderedDict([
                                          ("loss", f"{avg_losses['xh']:.4f}"),
                                          ("x_loss", f"{avg_losses['x']:.4f}"),
                                          ("h_loss", f"{avg_losses['h']:.4f}")
                                          ]))
        progress_bar.update()
    return avg_losses 

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/QM9_ddpm.yaml', help='Path to yaml configuration file (default: %(default)s)')
    parser.add_argument('--sample_after_train', action="store_true", help="Enable sampling after training (disabled by default)")
    args = parser.parse_args()

    cfg_path = args.cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Set seed for reproducibility
    set_seed(cfg["seed"])
    # Load dataset
    train_set, train_infos, _ = load_QM9_split(data_dir=cfg["data_dir"],
                                            split="train",
                                            load_info=True)

    # If specified run only on a subset of 
    if cfg["debug_samples"] is not None:
        debug_samples = cfg["debug_samples"]
        train_set = train_set[:debug_samples]
        print(f"ATTENTION! Running debug only on {debug_samples} samples (ignore if sampling)")
    
    # Istantiate model
    mol_size_hist = {int(size): count for size, count in train_infos["mol_size_hist"].items()}
    graph_size_dist = SizeDistribution(sizes=list(mol_size_hist.keys()),
                                       counts=list(mol_size_hist.values()))
    edge_index_builder = FullyConnectedEdgeIndex(graphs_sizes=list(mol_size_hist.keys()))
    
    model_dict = {"EGNNIsoform":EGNNIsoform}
    
    noise_network = model_dict[cfg["model"]["noise_net"]]
    noise_network = noise_network(atom_pos_dim=cfg["model"]["atom_pos_dim"],
                                  atom_feat_dim=cfg["model"]["atom_feat_dim"],
                                  t_dim=cfg["model"]["t_dim"],
                                  h_proj_dim=cfg["model"]["h_proj_dim"],
                                  t_proj_dim=cfg["model"]["t_proj_dim"],
                                  d_proj_dim=cfg["model"]["d_proj_dim"],
                                  message_dim=cfg["model"]["message_dim"],
                                  num_mp_rounds=cfg["model"]["num_mp_rounds"],
                                  activation_fn=nn.SiLU,
                                  conditioning=cfg["model"]["conditioning"],
                                  normalize_aggr=cfg["model"]["normalize_aggr"],
                                  norm_factor=cfg["model"]["norm_factor"])

    model = DDPM(beta_min=cfg["model"]["beta_min"],
                 beta_max=cfg["model"]["beta_max"],
                 T=cfg["model"]["T"],
                 noise_network=noise_network,
                 edge_index_builder=edge_index_builder,
                 graph_size_dist=graph_size_dist,
                 atom_pos_dim=cfg["model"]["atom_pos_dim"],
                 atom_feat_dim=cfg["model"]["atom_feat_dim"],
                 t_dim=cfg["model"]["t_dim"],
                 loss_weights=cfg["train"]["loss_weights"]).to(cfg["device"])

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
        warm_up_epochs = cfg["train"]["warm_up_epochs"]
        warm_up_scheduler = LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warm_up_epochs)
        cosine_annealing_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warm_up_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warm_up_scheduler, cosine_annealing_scheduler], milestones=[warm_up_epochs])
        
        run_path = cfg["run_path"]
        if not cfg["resume_ckpt"]:
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
            # Load loss file
            loss_file = os.path.join(run_path, "loss.json")
            with open(loss_file, "r") as f:
                loss_history = json.load(f)
            print(f"Resuming training from checkpoint {last_ckpt}")

        # Train model 
        train(model=model,
              optimizer=optimizer,
              scheduler=scheduler,
              data_loader=train_loader,
              num_epochs=num_epochs,
              ckpt_epochs=cfg["train"]["ckpt_epochs"],
              save_ckpt=save_ckpt,
              run_path=cfg["run_path"],
              start_epoch=start_epoch,
              loss_history=loss_history,
              device=cfg["device"])
        
        # Save final model
        model_path = os.path.join(cfg["run_path"], "model.pt")
        torch.save(obj=model.state_dict(), f=model_path)
        # Save a copy of the training configurations
        cfg_copy_path = os.path.join(run_path, "cfg_train.yaml")
        with open(cfg_copy_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        
        if args.sample_after_train:
            cfg["mode"] = "sample"

    if cfg["mode"] == "test":
        # Assure existance of model source
        if not os.path.exists(cfg["run_path"]):
            print(f"The run path {cfg['run_path']} does not exist")
            exit()
        
        # Load model parameters
        model_path = os.path.join(cfg["run_path"], "model.pt")
        state = torch.load(model_path, map_location=cfg["device"])
        if "model_state" in state:  
            model.load_state_dict(state["model_state"])
        else:  # pure state_dict
            model.load_state_dict(state)
        test_set, _, _= load_QM9_split(data_dir=cfg["data_dir"],
                                  split="test",
                                  load_info=False)
        # Istantiate dataloader
        test_loader = DataLoader(test_set, batch_size=cfg["test"]["batch_size"], shuffle=False)
        
        loss_history = {"xh": [], "x": [], "h": []}   
        num_test_repetitions = cfg["test"]["num_test_repetitions"]
        print(f"Testing the model for {num_test_repetitions} times")
        for i in range(num_test_repetitions):
            # compute and log loss of current repetition
            loss_rep = test(model, test_loader, cfg["device"])
            for key in loss_rep.keys():
                loss_history[key].append(loss_rep[key])

        print(f"{'Metric':<15} {'Mean':>15} {'Std':>15}")
        print("-" * 45)
        print(f"{'xh':<15} {np.mean(loss_history['xh']):>15.6f} {np.std(loss_history['xh']):>15.6f}")
        print(f"{'x':<15} {np.mean(loss_history['x']):>15.6f} {np.std(loss_history['x']):>15.6f}")
        print(f"{'h':<15} {np.mean(loss_history['h']):>15.6f} {np.std(loss_history['h']):>15.6f}")


    if cfg["mode"] == "sample":
        # Assure existance of model source
        if not os.path.exists(cfg["run_path"]):
            print(f"The run path {cfg['run_path']} does not exist")
            exit()
        
        # Load model parameters
        model_path = os.path.join(cfg["run_path"], "model.pt")
        state = torch.load(model_path, map_location=cfg["device"])
        if "model_state" in state:  
            model.load_state_dict(state["model_state"])
        else:  # pure state_dict
            model.load_state_dict(state)
        print(f"Loaded model from {model_path}")
        # Generate samples
        num_samples = cfg["sample"]["num_samples"]
        atom_decoder = np.array(cfg["sample"]["atom_decoder"]) 
        coords, atom_types, batch = model.sample(num_samples=num_samples,
                                   batch_size=cfg["sample"]["batch_size"],
                                   c=cfg["sample"]["c"],
                                   device=cfg["device"])
        # Convert to atom types
        atom_types = atom_types.argmax(dim=1).numpy()
        atom_types = atom_decoder[atom_types]
        # Write samples to file
        sample_path = os.path.join(cfg["run_path"], "samples.xyz")
        with open(sample_path, "w") as f:
            for sample_idx in range(num_samples):
                mask = (batch == sample_idx)
                num_atoms = mask.sum()
                coords_sample = coords[mask]
                atom_types_sample = atom_types[mask]
                f.write(f"{num_atoms}\n")
                f.write("\n")
                for i in range(num_atoms):
                    atom = atom_types_sample[i]
                    x, y, z = coords_sample[i].tolist()
                    f.write(f"{atom:<2}  {x:25.15f}  {y:25.15f}  {z:25.15f}\n")
        print(f"Samples saved at: {sample_path}")
