"""
geoLinker v8 — losses.py (NUMERICALLY SECURED EDITION)
Standardized chemistry loss with strict steric clash thresholds
and endpoint-only connection requirements.
Uses safe_dist to completely prevent coordinate gradient singularities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_dist(x, y, eps=1e-8):
    """
    Computes pairwise Euclidean distances safely without coordinate gradient singularities.
    x: [N, 3], y: [M, 3]
    """
    x_sq = torch.sum(x**2, dim=-1, keepdim=True) # [N, 1]
    y_sq = torch.sum(y**2, dim=-1, keepdim=True) # [M, 1]
    cross = torch.matmul(x, y.T)                 # [N, M]
    
    dists_sq = x_sq + y_sq.T - 2 * cross
    return torch.sqrt(torch.clamp(dists_sq, min=eps))

class DenoisingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_noise, true_noise, mask=None):
        if not torch.isfinite(pred_noise).all():
            return torch.tensor(0.0, device=pred_noise.device, requires_grad=True)
            
        loss = F.mse_loss(pred_noise, true_noise, reduction='none').mean(dim=-1)
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

class AtomTypeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weights', torch.tensor([1.0, 4.0, 4.0, 8.0, 8.0, 6.0, 8.0, 8.0]))
        
    def forward(self, pred_types, true_types, mask=None):
        loss = F.cross_entropy(pred_types, true_types, weight=self.weights, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

class StrictStericClashLoss(nn.Module):
    def __init__(self, clash_threshold: float = 1.2):
        super().__init__()
        self.clash_threshold = clash_threshold
        
    def forward(self, positions, batch_mask, pocket_pos=None, pocket_batch_mask=None):
        if not torch.isfinite(positions).all():
            return torch.tensor(0.0, device=positions.device)

        N_total = len(positions)
        batch_size = int(batch_mask.max()) + 1
        
        # 1. Standard same-molecule ligand clash (No loops!)
        distances = safe_dist(positions, positions) # [N_total, N_total]
        adj_mask = (batch_mask.unsqueeze(0) == batch_mask.unsqueeze(1)) # [N_total, N_total]
        
        # Ignore diagonal and cross-molecule interactions
        mask = adj_mask & (~torch.eye(N_total, dtype=torch.bool, device=distances.device))
        
        clashes = F.relu(self.clash_threshold - distances)
        clashes = clashes * mask.float()
        clashes = torch.clamp(clashes, max=5.0)
        total_loss = (clashes ** 2).sum()
        
        # 2. Active Pocket Clash Prevention (No loops!)
        if pocket_pos is not None and pocket_batch_mask is not None:
            pkt_distances = safe_dist(positions, pocket_pos) # [N_lig, N_pkt]
            adj_mask_pkt = (batch_mask.unsqueeze(1) == pocket_batch_mask.unsqueeze(0)) # [N_lig, N_pkt]
            
            pkt_clashes = F.relu(self.clash_threshold - pkt_distances)
            pkt_clashes = pkt_clashes * adj_mask_pkt.float()
            pkt_clashes = torch.clamp(pkt_clashes, max=5.0)
            total_loss += (pkt_clashes ** 2).sum()
            
        return total_loss / (batch_size + 1e-8)

class EndpointDistanceLoss(nn.Module):
    def __init__(self, target_distance: float = 1.5):
        super().__init__()
        self.target_distance = target_distance
        
    def forward(self, positions, linker_mask, anchor_mask, batch_mask):
        device = positions.device
        N_total = len(positions)
        batch_size = int(batch_mask.max()) + 1
        
        is_anchor = anchor_mask.bool()
        is_linker = linker_mask.bool()

        # BUG FIX (re-applied): this used to compare against fragment atoms,
        # which trivially includes the anchor atom itself at distance 0,
        # making this loss a constant with zero real gradient. Anchor atoms
        # must be measured against LINKER atoms -- the thing they're supposed
        # to bond to.
        if is_anchor.sum() == 0 or is_linker.sum() == 0:
            return torch.tensor(0.0, device=device)
            
        distances = safe_dist(positions, positions) # [N_total, N_total]
        adj_mask = (batch_mask.unsqueeze(0) == batch_mask.unsqueeze(1))
        
        # Keep only (anchor -> same-molecule-LINKER) pairs
        valid_pairs = adj_mask & is_anchor.unsqueeze(1) & is_linker.unsqueeze(0)
        
        # Mask out invalid pairs with a huge value so min() ignores them
        dists_masked = distances.masked_fill(~valid_pairs, 1e9)
        
        # Get, per anchor atom, the distance to its nearest linker atom
        min_dists, _ = dists_masked.min(dim=1)
        anchor_min_dists = min_dists[is_anchor]
        
        if len(anchor_min_dists) > 0:
            target = torch.ones_like(anchor_min_dists) * self.target_distance
            loss = F.smooth_l1_loss(anchor_min_dists, target)
            return loss
            
        return torch.tensor(0.0, device=device)


class NonAnchorRepulsionLoss(nn.Module):
    """
    Penalizes LINKER atoms landing within bonding distance of FRAGMENT atoms
    that are NOT designated anchor points AND not direct geometric neighbors
    of an anchor.

    Why this exists: atom_type_distribution.py confirmed the model's linker
    atom-type predictions are essentially perfect (match ground-truth
    frequencies almost exactly; halogens like Cl never even appear in linker
    atoms in this dataset). Yet generated samples were failing chemistry
    sanitization specifically on atoms like Chlorine -- which, since
    fragments are never generated/modified, can ONLY be a real FRAGMENT atom.
    So the bug isn't atom typing, it's geometric: linker atoms landing close
    enough to an already-saturated, non-anchor fragment atom (e.g. a terminal
    Cl substituent) to register as a spurious extra bond during
    reconstruction. Nothing previously penalized this -- StrictStericClashLoss's
    1.2 A threshold is a hard-overlap check, well inside real bonding
    distance (~1.5-1.8 A), so a linker atom sitting right at bonding range to
    a non-anchor fragment atom was invisible to every existing loss term.

    IMPORTANT refinement: a linker atom correctly bonding to the anchor at
    ~1.5 A (EndpointDistanceLoss's target) will often land within bonding
    range of the anchor's OWN real neighbors too, by simple bond-angle
    geometry -- fragments are small, so an anchor's neighbors are typically
    only ~1.5 A from the anchor itself. An earlier version of this loss
    penalized that directly, fighting EndpointDistanceLoss for every
    anchor-adjacent atom and measurably hurting connectivity/validity in
    practice. Fragment atoms within anchor_neighbor_exclude_distance of any
    anchor (in the fixed, never-moving fragment geometry) are now excluded
    from the penalty -- only fragment atoms that are genuinely far from every
    anchor are still off-limits.
    """
    def __init__(self, min_distance: float = 1.8, anchor_neighbor_exclude_distance: float = 1.9):
        super().__init__()
        self.min_distance = min_distance
        self.anchor_neighbor_exclude_distance = anchor_neighbor_exclude_distance

    def forward(self, positions, linker_mask, anchor_mask, batch_mask):
        device = positions.device
        is_linker = linker_mask.bool()
        is_frag = ~is_linker
        is_anchor = anchor_mask.bool()

        if is_linker.sum() == 0 or is_frag.sum() == 0:
            return torch.tensor(0.0, device=device)

        distances = safe_dist(positions, positions)
        adj_mask = (batch_mask.unsqueeze(0) == batch_mask.unsqueeze(1))

        if is_anchor.sum() > 0:
            anchor_dists = distances.masked_fill(~(adj_mask & is_anchor.unsqueeze(0)), 1e9)
            min_dist_to_anchor, _ = anchor_dists.min(dim=1)
            is_anchor_neighbor = (min_dist_to_anchor < self.anchor_neighbor_exclude_distance) & is_frag
        else:
            is_anchor_neighbor = torch.zeros_like(is_frag)

        is_offlimits_frag = is_frag & (~is_anchor) & (~is_anchor_neighbor)

        if is_offlimits_frag.sum() == 0:
            return torch.tensor(0.0, device=device)

        valid_pairs = adj_mask & is_linker.unsqueeze(1) & is_offlimits_frag.unsqueeze(0)

        violation = F.relu(self.min_distance - distances)
        violation = violation * valid_pairs.float()
        violation = torch.clamp(violation, max=5.0)
        batch_size = int(batch_mask.max()) + 1
        return (violation ** 2).sum() / (batch_size + 1e-8)

class UnifiedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_loss = DenoisingLoss()
        self.type_loss = AtomTypeLoss()
        self.clash_loss = StrictStericClashLoss()
        self.endpoint_loss = EndpointDistanceLoss()
        self.repulsion_loss = NonAnchorRepulsionLoss()
        
    def forward(self, outputs, batch):
        pred_noise = outputs['pred_noise']
        true_noise = outputs['true_noise']
        pred_types = outputs['pred_types']
        true_types = torch.argmax(batch['atom_features'], dim=1)
        
        linker_mask = batch['linker_mask']
        anchor_mask = batch.get('anchor_mask', torch.zeros_like(linker_mask))
        batch_mask = batch['batch_mask']
        
        est_positions = torch.clamp(outputs['est_positions'], min=-50.0, max=50.0)
        
        loss_denoise = self.denoise_loss(pred_noise, true_noise, linker_mask)
        loss_type = self.type_loss(pred_types, true_types, linker_mask)
        
        loss_clash = self.clash_loss(
            est_positions, 
            batch_mask, 
            batch.get('pocket_pos', None), 
            batch.get('pocket_batch_mask', None)
        )
        
        loss_endpoint = self.endpoint_loss(est_positions, linker_mask, anchor_mask, batch_mask)
        loss_repulsion = self.repulsion_loss(est_positions, linker_mask, anchor_mask, batch_mask)
        
        total_loss = (
            1.0 * loss_denoise +
            0.5 * loss_type +
            0.2 * loss_clash +
            0.4 * loss_endpoint +
            0.08 * loss_repulsion
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'denoise': loss_denoise.item(),
            'type': loss_type.item(),
            'clash': loss_clash.item(),
            'endpoint': loss_endpoint.item(),
            'repulsion': loss_repulsion.item()
        }
        
        return total_loss, loss_dict