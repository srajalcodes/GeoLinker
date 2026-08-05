"""
geoLinker v8 — models.py (SATORRAS-STABILIZED BACKBONE)
Implements a Hybrid EGNN + Self-Attention Backbone.
Uses a smooth Tanh coordinate update stabilizer to prevent SDE divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard transformer-style sinusoidal embedding for the diffusion timestep
    (Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020).
    Every diffusion network needs to know how noisy its current input is;
    without this, the model is forced to learn one noise-level-agnostic
    mapping, which cannot represent a correct reverse process.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device).float() / (half_dim - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class EGNN_Attention_Layer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Message network (MLP_e)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate update network (MLP_x)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        
        # Global Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Node update network (MLP_h)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

        # Gated Bidirectional Cross-Attention (CABF) for Pocket Conditioning (Vectorized)
        self.pocket_attention_lig = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.pocket_attention_pkt = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.gate_lig = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.gate_pkt = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(self, h: torch.Tensor, x: torch.Tensor, batch_mask: torch.Tensor,
                h_pkt: Optional[torch.Tensor] = None, x_pkt: Optional[torch.Tensor] = None, 
                batch_mask_pkt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        N = h.shape[0]
        batch_size = int(batch_mask.max()) + 1
        
        # 1. Adjacency mask for same-molecule pairs
        adj_mask = (batch_mask.unsqueeze(0) == batch_mask.unsqueeze(1)) # [N, N]
        
        # 2. Vectorized distance squared
        distances_sq = torch.sum((x.unsqueeze(1) - x.unsqueeze(0)) ** 2, dim=-1) # [N, N]
        distances_sq = distances_sq.masked_fill(~adj_mask, 0.0)
        
        # 3. Vectorized Message Passing
        h_i = h.unsqueeze(1).expand(-1, N, -1)
        h_j = h.unsqueeze(0).expand(N, -1, -1)
        edge_input = torch.cat([h_i, h_j, distances_sq.unsqueeze(-1)], dim=-1) # [N, N, 2*hidden + 1]
        
        m_ij = self.edge_mlp(edge_input) # [N, N, hidden_dim]
        m_ij = m_ij * adj_mask.unsqueeze(-1).float() # Zero out messages between different molecules
        
        # 4. Vectorized Coordinate Update with SMOOTH TANH STABILIZATION (Standard EGNN)
        # Tanh squashes coordinate updates smoothly to [-1.0, 1.0], preventing SDE explosions
        raw_weights = self.coord_mlp(m_ij).squeeze(-1)
        coord_weights = torch.tanh(raw_weights)
        coord_weights = coord_weights.masked_fill(~adj_mask, 0.0)
        
        coord_diffs = x.unsqueeze(1) - x.unsqueeze(0) # [N, N, 3]
        coord_update = torch.sum(coord_diffs * coord_weights.unsqueeze(-1), dim=1) # [N, 3]

        # FIX: normalize by each node's own same-molecule neighbor count, not the
        # total atom count of the whole batch. The old `/ (N + eps)` used N =
        # total atoms across ALL molecules in the batch, so the exact same
        # learned weights produced coordinate steps that scaled inversely with
        # batch_size -- e.g. ~32x larger at single-molecule inference time than
        # during batch_size=32 training. That mismatch alone is enough to blow
        # up coordinates over 6 layers x 50 diffusion steps. Normalizing by the
        # per-node neighbor count (Satorras et al., "E(n) Equivariant Graph
        # Neural Networks," ICML 2021) makes the update scale-consistent
        # regardless of how many molecules happen to share the batch.
        neighbor_counts = adj_mask.float().sum(dim=1, keepdim=True)  # [N, 1]
        x_new = x + coord_update / (neighbor_counts + 1e-8)
        
        # 5. Global Self-Attention with MASK BYPASS
        if batch_size > 1:
            h_attn, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), attn_mask=~adj_mask)
        else:
            h_attn, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
            
        h_norm = self.attn_norm(h + h_attn.squeeze(0))
        
        # 6. Gated Bidirectional Cross-Attention with Pocket (CABF) with MASK BYPASS
        if h_pkt is not None and x_pkt is not None and batch_mask_pkt is not None:
            adj_mask_pkt = (batch_mask.unsqueeze(1) == batch_mask_pkt.unsqueeze(0)) # [N_lig, N_pkt]
            
            # Pocket-to-Ligand Attention
            if batch_size > 1:
                attn_lig, _ = self.pocket_attention_lig(h_norm.unsqueeze(0), h_pkt.unsqueeze(0), h_pkt.unsqueeze(0), attn_mask=~adj_mask_pkt)
            else:
                attn_lig, _ = self.pocket_attention_lig(h_norm.unsqueeze(0), h_pkt.unsqueeze(0), h_pkt.unsqueeze(0))
                
            g_l = self.gate_lig(torch.cat([h_norm, attn_lig.squeeze(0)], dim=-1))
            h_norm = (1.0 - g_l) * h_norm + g_l * attn_lig.squeeze(0)
            
            # Ligand-to-Pocket Attention
            if batch_size > 1:
                attn_pkt, _ = self.pocket_attention_pkt(h_pkt.unsqueeze(0), h_norm.unsqueeze(0), h_norm.unsqueeze(0), attn_mask=~adj_mask_pkt.T)
            else:
                attn_pkt, _ = self.pocket_attention_pkt(h_pkt.unsqueeze(0), h_norm.unsqueeze(0), h_norm.unsqueeze(0))
                
            g_p = self.gate_pkt(torch.cat([h_pkt, attn_pkt.squeeze(0)], dim=-1))
            h_pkt = (1.0 - g_p) * h_pkt + g_p * attn_pkt.squeeze(0)
        
        # 7. Invariant Node Update
        m_i = torch.sum(m_ij, dim=1) # [N, hidden_dim]
        node_input = torch.cat([h_norm, m_i], dim=-1)
        h_new = self.node_norm(h_norm + self.node_mlp(node_input))
        
        return h_new, x_new

class GeoLinkerUnified(nn.Module):
    def __init__(self, num_atom_features: int = 8, hidden_dim: int = 128, num_layers: int = 6, max_size: int = 20):
        super().__init__()
        self.variant = 'unified'
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        
        self.atom_embed = nn.Linear(num_atom_features, hidden_dim)
        
        # Condition encoders
        self.anchor_encoder = nn.Sequential(
            nn.Linear(num_atom_features + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.anchor_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        self.size_embedding = nn.Embedding(max_size + 1, hidden_dim)
        self.size_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Pocket Encoder (for active cross-conditioning)
        self.pocket_embed = nn.Linear(num_atom_features, hidden_dim)
        
        # Diffusion timestep conditioning (was previously declared but never
        # wired into forward() -- see SinusoidalTimeEmbedding docstring)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Backbone Layers
        self.layers = nn.ModuleList([
            EGNN_Attention_Layer(hidden_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        
        # Output Heads
        self.type_head = nn.Linear(hidden_dim, num_atom_features)

    def forward(self, atom_features, positions, timestep, batch_mask, linker_mask, 
                anchor_mask=None, linker_size=None, pocket_pos=None, pocket_atoms=None, 
                pocket_batch_mask=None, **kwargs):
        
        h = self.atom_embed(atom_features)
        x = positions
        
        # Active Pocket embedding if provided
        h_pkt = None
        if pocket_pos is not None and pocket_atoms is not None:
            h_pkt = self.pocket_embed(pocket_atoms)
        
        if anchor_mask is not None and anchor_mask.sum() > 0:
            anchor_indicator = anchor_mask.unsqueeze(-1).float()
            anchor_input = torch.cat([atom_features, anchor_indicator], dim=-1)
            anchor_feats = self.anchor_encoder(anchor_input)
            
            is_linker = linker_mask.bool()
            is_anchor = anchor_mask.bool()
            if is_anchor.sum() > 0 and is_linker.sum() > 0:
                q = h[is_linker].unsqueeze(0)
                kv = anchor_feats[is_anchor].unsqueeze(0)
                attended_linker, _ = self.anchor_attention(q, kv, kv)
                h[is_linker] = h[is_linker] + attended_linker.squeeze(0)

        if linker_size is not None:
            size_embed = self.size_embedding(linker_size.clamp(0, self.max_size))
            size_features = self.size_proj(size_embed)
            h = h + size_features[batch_mask]

        # Timestep conditioning: broadcast the per-molecule time embedding to
        # every atom of that molecule via batch_mask, same pattern as size.
        t_embed = self.time_embed(timestep)
        t_features = self.time_proj(t_embed)
        h = h + t_features[batch_mask]

        # EGNN + Global Attention + Gated Bidirectional Pocket Cross-Attention
        for layer in self.layers:
            h, x = layer(
                h=h, 
                x=x, 
                batch_mask=batch_mask,
                h_pkt=h_pkt,
                x_pkt=pocket_pos,
                batch_mask_pkt=pocket_batch_mask
            )
            
        pred_atom_types = self.type_head(h)
        
        return {
            'pred_noise': x,             
            'pred_types': pred_atom_types,
            'est_positions': x
        }

def get_model(variant: str, **model_kwargs) -> nn.Module:
    return GeoLinkerUnified(**model_kwargs)