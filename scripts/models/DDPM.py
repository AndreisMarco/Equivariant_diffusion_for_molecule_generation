import torch
import torch.nn as nn
from torch_geometric.data import Data as GraphObject 
from scripts.utils import center_positions
from scripts.models.EGNN_Hoogeboom import EGNN as EGNN_Hoogeboom
from tqdm import tqdm

class DDPM(nn.Module):
    def __init__(self, noise_net,
                 T=1000, beta_min=1e-4, beta_max=0.02,  
                 input_dim=5, coord_dim=3, 
                 loss_weights={"h": 0.5, "x": 0.5},
                 edge_index_builder=None, graph_size_dist=None,
                 device="cuda"):
        super(DDPM, self).__init__()
        self.T = T
        self.register_buffer('beta', torch.linspace(beta_min, beta_max, T))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', self.alpha.cumprod(dim=0))
        self.coord_dim = coord_dim
        self.input_dim = input_dim
        self.edge_index_builder = edge_index_builder
        self.graph_size_dist = graph_size_dist
        self.loss_weights = loss_weights
        self.device = device
        self.noise_net = noise_net

    def loss(self, batched_graphs):
        batch = batched_graphs.batch
        num_graphs = int(batch.max().item()) + 1
        # sample timesteps
        t = torch.randint(0, self.T, (num_graphs,)).to(self.device)
        t = t.view(-1, 1)
        t = t[batch]
        # sample epsilon
        eps_x = torch.randn_like(batched_graphs.x)
        eps_x = center_positions(eps_x, batch)
        eps_h = torch.randn_like(batched_graphs.h)
        # compute noised z_t
        alpha_bar_t = self.alpha_bar[t]
        x_t = torch.sqrt(alpha_bar_t) * batched_graphs.x + torch.sqrt(1 - alpha_bar_t) * eps_x
        h_t = torch.sqrt(alpha_bar_t) * batched_graphs.h + torch.sqrt(1 - alpha_bar_t) * eps_h
        # predict noise
        fc_edge_index = self.edge_index_builder(batch)
        noised_graphs = GraphObject(x=x_t, h=h_t, batch=batch,
                                    edge_index=fc_edge_index,
                                    c=None)
        if isinstance(self.noise_net, EGNN_Hoogeboom):
            t = t/self.T
        eps_theta_x, eps_theta_h = self.noise_net(noised_graphs, t)
        eps_theta_x = center_positions(eps_theta_x, batch)
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
    def sample(self, num_samples, batch_size, c=None):
        # Sample n_samples sizes from the empirical size distribution of the training set
        num_nodes = self.graph_size_dist.sample(num_samples)
        total_nodes = torch.sum(num_nodes)
        # Initialize storing tensors
        all_h = torch.zeros(size=(total_nodes, self.input_dim), device=self.device)
        all_x = torch.zeros(size=(total_nodes, self.coord_dim), device=self.device)
        all_batch = torch.zeros(size=(total_nodes,), device=self.device)
        full_idx = 0 # Keeps track of how much of the tensors have been filled
        for i in tqdm(range(0, num_samples, batch_size), desc="Sampling"):
            num_nodes_curr = num_nodes[i:i + batch_size]
            total_nodes_curr = sum(num_nodes_curr)
            batch = torch.repeat_interleave(torch.arange(len(num_nodes_curr)),
                                            num_nodes_curr).to(self.device)
            # Create the edge index
            fc_edge_index = self.edge_index_builder(batch)
            # Sample zero CoM node pos and feats from normal prior
            x_t = torch.randn((total_nodes_curr, self.coord_dim), device=self.device)
            h_t = torch.randn((total_nodes_curr, self.input_dim), device=self.device)
            x_t = center_positions(x_t, batch)
            # Reverse diffusion
            for t in range(self.T - 1, -1, -1):
                t_batch = torch.full((total_nodes_curr, 1), t, device=self.device)
                if t > 0:
                    sigma = torch.sqrt(self.beta[t])
                    x_z = torch.randn_like(x_t)
                    h_z = torch.randn_like(h_t)
                else:
                    sigma = 0.0
                    x_z = torch.zeros_like(x_t)
                    h_z = torch.zeros_like(h_t)
                # Predict current step noise
                if isinstance(self.noise_net, EGNN_Hoogeboom):
                    t_batch = t_batch/self.T
                noised_graphs = GraphObject(x=x_t, h=h_t, batch=batch,
                                            edge_index=fc_edge_index,
                                            c=c)
                eps_theta_x, eps_theta_h = self.noise_net(noised_graphs, t_batch)
                eps_theta_x = center_positions(eps_theta_x, batch)
                # Compute one-step denoised latent
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                x_t = (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + 1e-8)) * eps_theta_x) \
                    / torch.sqrt(alpha_t + 1e-8) + sigma * x_z
                x_t = center_positions(pos=x_t, batch=batch)
                h_t = (h_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + 1e-8)) * eps_theta_h) \
                    / torch.sqrt(alpha_t + 1e-8) + sigma * h_z
            # Store denoised samples 
            all_x[full_idx : full_idx + total_nodes_curr] = x_t
            all_h[full_idx : full_idx + total_nodes_curr] = h_t
            all_batch[full_idx : full_idx + total_nodes_curr] = batch + i
            full_idx += total_nodes_curr
        return all_x.cpu(), all_h.cpu(), all_batch.cpu()
