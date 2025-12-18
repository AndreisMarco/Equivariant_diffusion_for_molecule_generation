import torch
import torch.nn as nn
import math

class EGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, out_dim=None,
                 num_layers=5,
                 act_fn=nn.SiLU(),
                 d_proj_dim=32, t_emb_dim=32,
                 agg_method="sum",
                 normalize_aggr=False, 
                 norm_factor=100,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.agg_method = agg_method
        self.normalize_aggr = normalize_aggr
        self.norm_factor = norm_factor
        self.t_embedding = SinusoidEmbedding(embedding_dim=t_emb_dim)
        
        if out_dim == None:
            out_dim = input_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_out = nn.Linear(hidden_dim, out_dim)
        message_input_dim = hidden_dim * 2 + d_proj_dim

        self.h_projection_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim))
        
        self.t_projection_net = nn.Sequential(
            nn.Linear(t_emb_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim))

        self.dist_projection_net = nn.Sequential(
            nn.Linear(1, d_proj_dim),
            act_fn,
            nn.Linear(d_proj_dim, d_proj_dim))
            
        layer = nn.Linear(hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.scalar_weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            layer,
            nn.Tanh())

        self.message_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(message_input_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, hidden_dim),
               act_fn,
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)])

        # self.update_net = nn.ModuleList([
        #     nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     ) for _ in range(num_layers)])
        
        self.out_network = nn.Linear(hidden_dim, input_dim)

    def forward(self, batched_graphs, t):
        x = batched_graphs.x
        h = batched_graphs.h
        row, col = batched_graphs.edge_index
        norm_factor = 1.0
        if self.normalize_aggr:    
            if self.norm_factor == "neighbours":
                norm_factor = torch.bincount(row).unsqueeze(1).float() # Note: given fc graphs this is always graph_size
            else:
                norm_factor = self.norm_factor

        t = self.t_embedding(t)
        t = self.t_projection_net(t)
        h = self.h_projection_net(h)
        for r in range(self.num_layers):
            sq_dist = ((x[row] - x[col]) ** 2).sum(dim=1, keepdim=True)
            dist = torch.sqrt(sq_dist + 1e-10)
            d = self.dist_projection_net(dist)
            message_net_input = torch.cat([h[row] + t[row], h[col] + t[col], d], dim=1)
            messages = self.message_net[r](message_net_input) 

            # position update 
            scalar_weights = self.scalar_weight_net(messages)
            pos_updates = (x[row] - x[col])*scalar_weights 
            aggr_pos_updates = agg_fn(index=row, source=pos_updates,
                                      agg_method=self.agg_method, 
                                      norm_factor=norm_factor)
            x = x + aggr_pos_updates 

            # feature update
            aggr_messages = agg_fn(index=row, source=messages, 
                                   agg_method=self.agg_method,
                                   norm_factor=norm_factor)
            # h = h + self.update_net[r](torch.cat([h, aggr_messages], dim=1))
            h = h + aggr_messages
        h = self.out_network(h)
        return x, h
    
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
        results = results / norm
    return results

class SinusoidEmbedding(nn.Module):
    def __init__(self, norm_factor=10000, embedding_dim=32):
        '''
        Sinusoidal embeddings from "Attention Is All You Need"
        '''
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