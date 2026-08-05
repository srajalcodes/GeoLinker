"""
geoLinker v8 — diffusion.py (PAPER-ALIGNED SDE)
Implements Centroid-Interpolated Initialization (Eq S4) and fixed fragment CoM projection.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

def zero_com_(noise: torch.Tensor, active_mask: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
    """
    Projects `noise` onto the zero-center-of-mass subspace.
    To prevent fragment-linker translation drift, this must project the ENTIRE molecule's CoM,
    or specifically respect the fixed fragment frame.
    """
    out = noise.clone()
    active_idx = active_mask.bool()
    if active_idx.sum() == 0:
        return out
    
    b = batch_mask[active_idx]
    active_noise = noise[active_idx]
    
    n_mol = int(batch_mask.max().item()) + 1
    sums = torch.zeros(n_mol, noise.shape[-1], device=noise.device, dtype=noise.dtype)
    counts = torch.zeros(n_mol, device=noise.device, dtype=noise.dtype)
    
    sums.index_add_(0, b, active_noise)
    counts.index_add_(0, b, torch.ones_like(b, dtype=noise.dtype))
    
    means = sums / counts.clamp(min=1).unsqueeze(-1)
    out[active_idx] = active_noise - means[b]
    return out

class EquivariantDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_schedule: str = 'cosine', variant: str = 'unified'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.variant = variant
        
        betas = self._cosine_beta_schedule(timesteps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        eps = 1e-8
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=alphas.device), alphas_cumprod[:-1]])
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod + eps))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod + eps))
        
        variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + eps)
        self.register_buffer('posterior_variance', torch.clamp(variance, min=1e-8))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha_cumprod_t.unsqueeze(-1) * x_start + sqrt_one_minus_t.unsqueeze(-1) * noise, noise

    def training_step(self, batch: Dict) -> Dict:
        device = batch['positions'].device
        batch_mask = batch['batch_mask']
        batch_size = int(batch_mask.max()) + 1
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        x_start = batch['positions']
        linker_mask = batch['linker_mask']
        anchor_mask = batch.get('anchor_mask', torch.zeros_like(linker_mask))

        noise = torch.randn_like(x_start)
        noise = zero_com_(noise, linker_mask, batch_mask)
        z_t, _ = self.q_sample(x_start, t[batch_mask], noise)
        
        preserve_mask = (1.0 - linker_mask).unsqueeze(-1)
        x_t = x_start * preserve_mask + z_t * (1.0 - preserve_mask)

        outputs = self.model(
            atom_features=batch['atom_features'],
            positions=x_t,
            timestep=t,
            batch_mask=batch_mask,
            linker_mask=linker_mask,
            anchor_mask=anchor_mask,
            linker_size=batch.get('linker_size', None),
            pocket_pos=batch.get('pocket_pos', None),
            pocket_atoms=batch.get('pocket_atoms', None),
            pocket_batch_mask=batch.get('pocket_batch_mask', None)
        )

        return {
            'pred_noise': outputs['pred_noise'], 
            'true_noise': x_start,               
            'pred_types': outputs['pred_types'],
            'est_positions': outputs['est_positions']
        }

    def _respaced_posterior(self, steps_sequence, device):
        eps = 1e-8
        ac = self.alphas_cumprod
        coefs = {}
        for i, t in enumerate(steps_sequence):
            t_prev = steps_sequence[i + 1] if i + 1 < len(steps_sequence) else -1
            ac_t = ac[t]
            ac_prev = ac[t_prev] if t_prev >= 0 else torch.ones((), device=device)
            beta_eff = 1.0 - ac_t / ac_prev.clamp(min=eps)
            coef1 = beta_eff * torch.sqrt(ac_prev) / (1.0 - ac_t + eps)
            coef2 = (1.0 - ac_prev) * torch.sqrt((ac_t / ac_prev.clamp(min=eps)).clamp(min=0.0)) / (1.0 - ac_t + eps)
            variance = torch.clamp(beta_eff * (1.0 - ac_prev) / (1.0 - ac_t + eps), min=eps)
            coefs[t] = (coef1, coef2, variance)
        return coefs

    def apply_pocket_repulsion(self, curr_pos, pocket_pos, linker_mask, strength=0.1):
        """Active heuristic: Push generated linker coordinates away from pocket boundaries."""
        curr_pos = curr_pos.clone()
        linker_idx = linker_mask.bool()
        
        linker_coords = curr_pos[linker_idx]
        diffs = linker_coords.unsqueeze(1) - pocket_pos.unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)
        clashes = dists < 1.5
        
        if clashes.any():
            masked_diffs = diffs * clashes.unsqueeze(-1).float()
            push_sum = masked_diffs.sum(dim=1)
            counts = clashes.sum(dim=1, keepdim=True)
            counts_safe = torch.clamp(counts, min=1)
            push_mean = push_sum / counts_safe
            push_norm = torch.norm(push_mean, dim=-1, keepdim=True) + 1e-8
            push_direction = push_mean / push_norm
            displacement = strength * push_direction * (counts > 0).float()
            curr_pos[linker_idx] += displacement
            
        return curr_pos

    @torch.no_grad()
    def sample(self, data_item: Dict, num_steps: int = 50, sample_strategy: str = 'ddpm') -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        
        atom_features = data_item['atom_features'].float().to(device)
        positions = data_item['positions'].float().to(device)
        linker_mask = data_item['linker_mask'].float().to(device)
        anchor_mask = data_item.get('anchor_mask', torch.zeros_like(linker_mask)).to(device)
        batch_mask = torch.zeros(len(positions), dtype=torch.long).to(device)
        
        N_link = int(linker_mask.sum().item())
        linker_size = torch.tensor([data_item.get('linker_size', N_link)], dtype=torch.long).to(device)

        pocket_pos = data_item.get('pocket_pos', None)
        pocket_atoms = data_item.get('pocket_atoms', None)
        if pocket_pos is not None:
            pocket_pos = pocket_pos.to(device)
            pocket_atoms = pocket_atoms.to(device)
        pocket_batch_mask = torch.zeros(len(pocket_pos), dtype=torch.long).to(device) if pocket_pos is not None else None

        curr_pos = positions.clone()
        linker_idx = linker_mask.bool()

        # ====================================================================
        # PAPER ALIGNMENT: Centroid-Interpolated Initialization (Equation S4)
        # ====================================================================
        # Identify fragments dynamically based on connectivity/distance
        frag_idx = ~linker_idx
        frag_pos = positions[frag_idx]
        
        if len(frag_pos) > 0 and N_link > 0:
            # Heuristic to separate the two fragments using KMeans or spatial splitting
            # Since fragments are typically spatially separated, we split by the longest axis
            max_dist_idx = torch.argmax(torch.cdist(frag_pos, frag_pos))
            idx1 = max_dist_idx // len(frag_pos)
            idx2 = max_dist_idx % len(frag_pos)
            
            p1, p2 = frag_pos[idx1], frag_pos[idx2]
            dists_to_p1 = torch.norm(frag_pos - p1, dim=-1)
            dists_to_p2 = torch.norm(frag_pos - p2, dim=-1)
            
            frag1_mask = dists_to_p1 < dists_to_p2
            c1 = frag_pos[frag1_mask].mean(dim=0)
            c2 = frag_pos[~frag1_mask].mean(dim=0)
            
            # Interpolate coordinates
            interpolated_coords = []
            for k in range(1, N_link + 1):
                t_k = k / (N_link + 1)
                eps = torch.randn(3, device=device) * 0.1  # 0.1 variance noise from paper
                pos_k = (1 - t_k) * c1 + t_k * c2 + eps
                interpolated_coords.append(pos_k)
            
            init_noise = torch.stack(interpolated_coords)
        else:
            # Fallback if no valid fragments
            init_noise = torch.randn((N_link, 3), device=device)
            
        curr_pos[linker_idx] = init_noise

        step_size = self.timesteps // num_steps
        steps_sequence = list(reversed(range(0, self.timesteps, step_size)))
        respaced = self._respaced_posterior(steps_sequence, device)

        for step in steps_sequence:
            t = torch.full((1,), step, dtype=torch.long, device=device)
            
            outputs = self.model(
                atom_features=atom_features,
                positions=curr_pos,
                timestep=t,
                batch_mask=batch_mask,
                linker_mask=linker_mask,
                anchor_mask=anchor_mask,
                linker_size=linker_size,
                pocket_pos=pocket_pos,
                pocket_atoms=pocket_atoms,
                pocket_batch_mask=pocket_batch_mask
            )
            
            x_0 = outputs['est_positions']
            coef1, coef2, variance = respaced[step]
            posterior_mean = coef1.unsqueeze(-1) * x_0 + coef2.unsqueeze(-1) * curr_pos
            
            if step > 0:
                noise = torch.randn_like(curr_pos)
                noise = zero_com_(noise, linker_mask, batch_mask)
                curr_pos = posterior_mean + torch.sqrt(variance.clamp(min=1e-8)).unsqueeze(-1) * noise
            else:
                curr_pos = posterior_mean

            if pocket_pos is not None:
                curr_pos = self.apply_pocket_repulsion(curr_pos, pocket_pos, linker_mask)

            # Apply Fragment Attraction Heuristic (Paper Note 3: Final 1/3rd of steps)
            if step < (self.timesteps // 3) and len(frag_pos) > 0 and N_link > 0:
                linker_coords = curr_pos[linker_idx]
                dists = torch.cdist(linker_coords, frag_pos)
                min_dists, nearest_frag_idx = dists.min(dim=1)
                
                # Nudge atoms that are too far (> 3.0 Å)
                too_far = min_dists > 3.0
                if too_far.any():
                    vectors_to_frag = frag_pos[nearest_frag_idx[too_far]] - linker_coords[too_far]
                    directions = vectors_to_frag / (torch.norm(vectors_to_frag, dim=1, keepdim=True) + 1e-8)
                    curr_pos[linker_idx][too_far] += 0.05 * directions # Strength 0.05 from paper

            # Strict Inpainting Constraint
            curr_pos[~linker_idx] = positions[~linker_idx]
            
            curr_pos = torch.nan_to_num(curr_pos, nan=0.0, posinf=1e2, neginf=-1e2)
            curr_pos = torch.clamp(curr_pos, min=-1e2, max=1e2)

        final_types = torch.argmax(outputs['pred_types'], dim=-1)
        return curr_pos, final_types