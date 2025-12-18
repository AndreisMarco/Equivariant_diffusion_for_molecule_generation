
import os
import json
import re

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch_geometric.datasets import QM9
from torch_geometric.utils import dense_to_sparse

class SizeDistribution:
    '''
    Provide sampling of graph sizes from a categorical distribution
    provided during initialization
    '''
    def __init__(self, sizes, counts):
        self.sizes = torch.as_tensor(sizes, dtype=torch.int)
        counts = torch.as_tensor(counts, dtype=torch.float)
        probs = counts / counts.sum()
        self.distribution = torch.distributions.Categorical(probs=probs)

    def sample(self, n_samples):
        idx = self.distribution.sample((n_samples,))
        return self.sizes[idx]
    
class FullyConnectedEdgeIndex:
    '''
    Provides precomputed fully connected edge_index matrices based on graph size 
    '''
    def __init__(self, graphs_sizes):
        self.precomputed = {}
        for num_nodes in graphs_sizes:
            num_nodes = int(num_nodes)
            adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            edge_index, _ = dense_to_sparse(adj)
            self.precomputed[num_nodes] = edge_index

    def __call__(self, batch):
        edge_indices = []
        cum_nodes_count = 0
        for graph_id in range(batch.max() + 1):
            node_mask = (batch == graph_id)
            n_nodes = node_mask.sum().item()
            edge_index = self.precomputed[n_nodes] + cum_nodes_count
            edge_indices.append(edge_index)
            cum_nodes_count += n_nodes
        return torch.cat(edge_indices, dim=1).to(batch.device)
    
class SinusoidalEmbedding(nn.Module):
    '''
    Sinusoidal embeddings from Vaswani et al. 2017
    '''
    def __init__(self, norm_factor=10000, embedding_dim=8):
        super().__init__()
        self.norm_factor = norm_factor
        assert embedding_dim % 2 == 0, f"Sinusoidal embeddings must have even dimension, got {embedding_dim}"
        self.embedding_dim = embedding_dim 
        i = torch.arange(embedding_dim // 2).view(-1,)
        self.register_buffer('freq', norm_factor ** (2 * i / embedding_dim))

    def forward(self, t):
        t = t.view(-1,1)
        sin_emb = torch.sin(t/self.freq)
        cos_emb = torch.cos(t/self.freq)
        embedding = torch.cat([sin_emb, cos_emb], dim=1)
        return embedding
    
def center_positions(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = batch.max().item() + 1
    pos_sum = torch.zeros(num_graphs, pos.size(1), dtype=pos.dtype, device=pos.device)
    pos_sum.index_add_(0, batch, pos)
    ones_tensor = torch.ones(pos.size(0), 1, dtype=pos.dtype, device=pos.device)
    graph_counts = torch.zeros(num_graphs, 1, dtype=pos.dtype, device=pos.device)
    graph_counts.index_add_(0, batch, ones_tensor)
    pos_mean = pos_sum / graph_counts
    pos_centered = pos - pos_mean[batch]
    return pos_centered

def save_and_plot_loss(loss_history, save_path):
    # Plot losses
    num_epochs = len(loss_history["xh"])
    plt.figure(figsize=(8, 5))
    for key in loss_history:
        plt.plot(range(num_epochs), loss_history[key], label=key, alpha=0.5)
    # Plot adjustments
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("DDPM Training Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    # Save plot and loss history
    plot_path = os.path.join(save_path, "loss.png")
    file_path = os.path.join(save_path, "loss.json")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    with open(file_path, "w") as f:
        json.dump(loss_history, f)

def write_xyz(atom_type, coord, batch, file):
    num_molecules = int(max(batch) + 1)
    with open(file, "w") as f:
        for i in range(num_molecules):
            mask = (batch == i)
            num_atoms = mask.sum()
            atom_type_mol = atom_type[mask]
            coord_mol = coord[mask]

            # Write xyz format
            f.write(f"{num_atoms}\n") # num atoms
            f.write("\n") # empty description
            for i in range(num_atoms): # atom type and coord
                atom = atom_type_mol[i]
                x, y, z = coord_mol[i].tolist()
                f.write(f"{atom:<2}  {x:25.15f}  {y:25.15f}  {z:25.15f}\n") 

def raw_out_log_to_loss_file(path):
    '''
    Extracts and saves losses from tqdm-style
    training logs written to a file.
    Keeps only the first occurrence per epoch.
    '''
    output_file = "loss.json"

    # Regex to extract epoch number and loss values
    pattern = re.compile(
        r"\|\s*(\d+)/\d+\s*\[.*?loss=([\d.]+),\s*x_loss=([\d.]+),\s*h_loss=([\d.]+)"
    )

    xh, x, h = [], [], []
    seen_epochs = set()

    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                if epoch in seen_epochs:
                    continue
                seen_epochs.add(epoch)

                xh.append(float(match.group(2)))
                x.append(float(match.group(3)))
                h.append(float(match.group(4)))

    data = {"xh": xh, "x": x, "h": h}

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(xh)} unique epochs to {output_file}")
