
import torch
import torch.nn as nn
import math

class FeatUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim,
                 act_fn, attention,
                 agg_method, norm_factor):
        super(FeatUpdate, self).__init__()
        self.agg_method = agg_method
        self.norm_factor = norm_factor
        self.attention = attention

        self.message_net = nn.Sequential(
            nn.Linear(input_dim * 2 + edge_attr_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, input_dim))
        
        if self.attention:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid())

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        # Compute messages between nodes
        messages = self.message_net(input_tensor)
        # Compute and apply attention weights
        if self.attention:
            attention_weights = self.attention_net(messages)
            messages = messages * attention_weights
        # aggegate messages
        agg = agg_fn(index=row, source=messages, 
                     agg_method=self.agg_method, norm_factor=self.norm_factor)
        # Update feature with skip connection
        h = h + self.update_net(torch.cat([h, agg], dim=1))
        return h
    
class CoordUpdate(nn.Module):
    def __init__(self, hidden_dim, edge_attr_dim,
                 act_fn, tanh, coords_range,
                 agg_method, norm_factor):
        super(CoordUpdate, self).__init__()
        self.agg_method = agg_method
        self.norm_factor = norm_factor
        self.tanh = tanh
        self.coords_range = coords_range

        layer = nn.Linear(hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.scalar_weight = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_attr_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            layer)

    def forward(self, h, coord, edge_index, coord_diff, edge_attr):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        # Computes coordinates updates
        if self.tanh:
            trans = coord_diff * torch.tanh(self.scalar_weight(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.scalar_weight(input_tensor)
        # Aggregate updates
        agg = agg_fn(index=row, source=trans,
                     agg_method=self.agg_method, norm_factor=self.norm_factor)
        # Update coordinates
        coord = coord + agg
        return coord

class UpdateBlock(nn.Module):
    '''
    Updates both coordinates and features
    '''
    def __init__(self, hidden_dim, edge_attr_dim, 
                 num_layers, attention, act_fn, tanh, coords_range,
                 agg_method, norm_factor,
                 norm_constant, sin_embedding,
                 device):
        super(UpdateBlock, self).__init__()
        self.num_layers = num_layers
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding

        for i in range(0, num_layers):
            self.add_module("gcl_%d" % i, FeatUpdate(input_dim=hidden_dim, hidden_dim=hidden_dim, edge_attr_dim=edge_attr_dim,
                                                     act_fn=act_fn, attention=attention,
                                                     agg_method=agg_method, norm_factor=norm_factor))
        self.add_module("gcl_equiv", CoordUpdate(hidden_dim=hidden_dim, edge_attr_dim=edge_attr_dim,
                                                 act_fn=act_fn, tanh=tanh, coords_range=coords_range,
                                                 agg_method=agg_method, norm_factor=norm_factor))
        self.to(device)

    def forward(self, h, x, edge_index, edge_attr):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        # Add current distances to distances from start of forward pass
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)

        for i in range(0, self.num_layers):
            h = self._modules["gcl_%d" % i](h=h, edge_index=edge_index, edge_attr=edge_attr)
        x = self._modules["gcl_equiv"](h=h, coord=x, edge_index=edge_index, coord_diff=coord_diff, edge_attr=edge_attr)
        return h, x
    
class EGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, out_dim=None,
                 num_layers=5, num_sublayers=1, attention=True,
                 act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 agg_method="sum", norm_factor=100,
                 norm_constant=1, sin_embedding=False,
                 device="cuda", **kwargs):
        super(EGNN, self).__init__()
        if out_dim == None:
            out_dim = input_dim
        self.num_layers = num_layers
        input_dim = input_dim + 1 #  add time dimension

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbedding()
            edge_attr_dim = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_attr_dim = 2

        self.coords_range_layer = float(coords_range/num_layers)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_out = nn.Linear(hidden_dim, out_dim)
        for i in range(0, num_layers):
            self.add_module("e_block_%d" % i, UpdateBlock(hidden_dim=hidden_dim, edge_attr_dim=edge_attr_dim,
                                                          num_layers=num_sublayers, attention=attention,
                                                          act_fn=act_fn, tanh=tanh, coords_range=self.coords_range_layer,
                                                          agg_method=agg_method, norm_factor=norm_factor,
                                                          norm_constant=norm_constant, sin_embedding=self.sin_embedding,
                                                          device=device))
        self.to(device)

    def forward(self, batched_graphs, t):
        x = batched_graphs.x
        h = torch.cat([batched_graphs.h, t], dim=1)
        edge_index = batched_graphs.edge_index
        # Use distance between nodes as edge_attr
        edge_attr, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            edge_attr = self.sin_embedding(edge_attr)
        # Project h to hidden dim
        h = self.embedding(h)
        # Update h and x
        for i in range(0, self.num_layers):
            h, x = self._modules["e_block_%d" % i](h=h, x=x, edge_index=edge_index, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return x, h
    
def coord2diff(coord, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]
    radial = torch.sum((coord_diff ** 2), 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff
    
def agg_fn(index, source, agg_method, norm_factor):
    result_shape = torch.Size([len(torch.unique(index)), source.shape[1]])
    results = torch.zeros(size=result_shape, device=index.device)
    results.index_add_(dim=0, index=index, source=source)

    if agg_method == "sum":
        results = results / norm_factor
    
    elif agg_method == "mean":
        norm = torch.zeros_like(results)
        norm.index_add_(dim=0, index=index, source=torch.ones_like(results))
        norm[norm == 0] = 1
        result = result / norm
    return results

class SinusoidsEmbedding(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()